import copy

import numpy as np

from onpolicy.envs.pso.functions import DEFAULT_BOUNDS, OBJECTIVES

try:
    import gym
except ImportError:
    class Box(object):
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class Discrete(object):
        def __init__(self, n):
            self.n = n

    class _Spaces(object):
        Box = Box
        Discrete = Discrete

    class _Gym(object):
        spaces = _Spaces()

    gym = _Gym()


class RingTopologyPSOEnv(object):
    """Constricted ring-neighborhood PSO as a cooperative multi-agent environment.

    One environment step is one PSO generation. Each particle is one agent and
    chooses one leader from {left neighbor, self, right neighbor}.
    """

    metadata = {"render.modes": ["human"]}

    ACTION_LEFT = 0
    ACTION_SELF = 1
    ACTION_RIGHT = 2

    def __init__(self, args):
        self.num_particles = int(args.pso_particles)
        self.dimension = int(args.pso_dim)
        self.max_generations = int(args.pso_generations)
        self.objective_name = args.pso_objective
        if self.objective_name not in OBJECTIVES:
            raise ValueError("Unknown PSO objective: {}".format(self.objective_name))
        self.objective = OBJECTIVES[self.objective_name]

        default_low, default_high = DEFAULT_BOUNDS[self.objective_name]
        lower_bound = getattr(args, "pso_lower_bound", default_low)
        upper_bound = getattr(args, "pso_upper_bound", default_high)
        self.lower_bound = float(default_low if lower_bound is None else lower_bound)
        self.upper_bound = float(default_high if upper_bound is None else upper_bound)
        self.chi = float(args.pso_chi)
        self.c1 = float(args.pso_c1)
        self.c2 = float(args.pso_c2)
        self.boundary = args.pso_boundary
        self.scale_floor = float(args.pso_scale_floor)
        self.initial_velocity_scale = float(args.pso_initial_velocity_scale)

        self.obs_dim = 17
        obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32)
        share_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.obs_dim * self.num_particles,),
            dtype=np.float32,
        )
        act_space = gym.spaces.Discrete(3)
        self.observation_space = [obs_space for _ in range(self.num_particles)]
        self.share_observation_space = [share_space for _ in range(self.num_particles)]
        self.action_space = [act_space for _ in range(self.num_particles)]

        self._seed = None
        self.rng = np.random.default_rng()
        self.reset()

    def seed(self, seed=None):
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        span = self.upper_bound - self.lower_bound
        self.generation = 0
        self.extra_evaluations = 0
        self.positions = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(self.num_particles, self.dimension),
        ).astype(np.float64)
        self.velocities = self.rng.uniform(
            -span * self.initial_velocity_scale,
            span * self.initial_velocity_scale,
            size=(self.num_particles, self.dimension),
        ).astype(np.float64)
        self.current_values = self._evaluate(self.positions)
        self.total_evaluations = self.num_particles
        self.pbest_positions = self.positions.copy()
        self.pbest_values = self.current_values.copy()
        self.pbest_age = np.zeros(self.num_particles, dtype=np.int64)
        self.stagnation = np.zeros(self.num_particles, dtype=np.int64)
        self.memory_version = np.zeros(self.num_particles, dtype=np.int64)
        self.adopted_counts = np.zeros(self.num_particles, dtype=np.int64)
        self.last_improvements = np.zeros(self.num_particles, dtype=np.float64)
        self.last_leaders = np.arange(self.num_particles, dtype=np.int64)
        self.last_leader_versions = self.memory_version.copy()
        self.initial_best_value = float(np.min(self.pbest_values))
        self.global_best_index = int(np.argmin(self.pbest_values))
        self.global_best_value = float(self.pbest_values[self.global_best_index])
        self.global_best_position = self.pbest_positions[self.global_best_index].copy()
        self.reward_scale = max(abs(self.initial_best_value), self.scale_floor)
        return self._get_obs()

    def step(self, actions):
        actions = self._decode_actions(actions)
        best_before = float(self.global_best_value)
        leader_indices = self._leader_indices(actions)
        self.last_leaders = leader_indices.copy()
        self.last_leader_versions = self.memory_version[leader_indices].copy()
        for leader_idx in leader_indices:
            self.adopted_counts[leader_idx] += 1

        r1 = self.rng.random(size=(self.num_particles, self.dimension))
        r2 = self.rng.random(size=(self.num_particles, self.dimension))
        cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
        social = self.c2 * r2 * (self.pbest_positions[leader_indices] - self.positions)
        self.velocities = self.chi * (self.velocities + cognitive + social)
        self.positions = self.positions + self.velocities
        self._apply_boundary()

        self.current_values = self._evaluate(self.positions)
        self.total_evaluations += self.num_particles
        improved = self.current_values < self.pbest_values
        previous_pbest = self.pbest_values.copy()
        self.pbest_positions[improved] = self.positions[improved]
        self.pbest_values[improved] = self.current_values[improved]
        self.last_improvements[:] = 0.0
        self.last_improvements[improved] = previous_pbest[improved] - self.current_values[improved]
        self.memory_version[improved] += 1
        self.pbest_age += 1
        self.pbest_age[improved] = 0
        self.stagnation += 1
        self.stagnation[improved] = 0

        self.global_best_index = int(np.argmin(self.pbest_values))
        self.global_best_value = float(self.pbest_values[self.global_best_index])
        self.global_best_position = self.pbest_positions[self.global_best_index].copy()
        reward = (best_before - self.global_best_value) / self.reward_scale

        self.generation += 1
        done = self.generation >= self.max_generations
        obs = self._get_obs()
        rewards = np.full((self.num_particles, 1), reward, dtype=np.float32)
        dones = np.full(self.num_particles, done, dtype=bool)
        infos = [self._agent_info(i, actions[i], leader_indices[i]) for i in range(self.num_particles)]
        return obs, rewards, dones, infos

    def get_snapshot(self):
        return {
            "generation": self.generation,
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "current_values": self.current_values.copy(),
            "total_evaluations": self.total_evaluations,
            "extra_evaluations": self.extra_evaluations,
            "pbest_positions": self.pbest_positions.copy(),
            "pbest_values": self.pbest_values.copy(),
            "pbest_age": self.pbest_age.copy(),
            "stagnation": self.stagnation.copy(),
            "memory_version": self.memory_version.copy(),
            "adopted_counts": self.adopted_counts.copy(),
            "last_improvements": self.last_improvements.copy(),
            "last_leaders": self.last_leaders.copy(),
            "last_leader_versions": self.last_leader_versions.copy(),
            "initial_best_value": self.initial_best_value,
            "global_best_index": self.global_best_index,
            "global_best_value": self.global_best_value,
            "global_best_position": self.global_best_position.copy(),
            "reward_scale": self.reward_scale,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore_snapshot(self, snapshot):
        self.generation = int(snapshot["generation"])
        self.positions = snapshot["positions"].copy()
        self.velocities = snapshot["velocities"].copy()
        self.current_values = snapshot["current_values"].copy()
        self.total_evaluations = int(snapshot["total_evaluations"])
        self.extra_evaluations = int(snapshot["extra_evaluations"])
        self.pbest_positions = snapshot["pbest_positions"].copy()
        self.pbest_values = snapshot["pbest_values"].copy()
        self.pbest_age = snapshot["pbest_age"].copy()
        self.stagnation = snapshot["stagnation"].copy()
        self.memory_version = snapshot["memory_version"].copy()
        self.adopted_counts = snapshot["adopted_counts"].copy()
        self.last_improvements = snapshot["last_improvements"].copy()
        self.last_leaders = snapshot["last_leaders"].copy()
        self.last_leader_versions = snapshot["last_leader_versions"].copy()
        self.initial_best_value = float(snapshot["initial_best_value"])
        self.global_best_index = int(snapshot["global_best_index"])
        self.global_best_value = float(snapshot["global_best_value"])
        self.global_best_position = snapshot["global_best_position"].copy()
        self.reward_scale = float(snapshot["reward_scale"])
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])

    def rollout_alternative_from_snapshot(self, snapshot, initial_actions, policy_fn):
        """Run one counterfactual branch and leave restoration to the caller."""
        self.restore_snapshot(snapshot)
        evals_before = self.total_evaluations
        obs, _, dones, _ = self.step(initial_actions)
        while not np.all(dones):
            actions = policy_fn(obs)
            obs, _, dones, _ = self.step(actions)
        branch_evals = self.total_evaluations - evals_before
        return float(self.global_best_value), int(branch_evals)

    def ring_best_actions(self):
        actions = np.zeros(self.num_particles, dtype=np.int64)
        for i in range(self.num_particles):
            candidates = self._candidate_indices(i)
            best_local = int(np.argmin(self.pbest_values[candidates]))
            actions[i] = best_local
        return actions

    def random_actions(self):
        return self.rng.integers(0, 3, size=self.num_particles, dtype=np.int64)

    def close(self):
        pass

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError
        print(
            "generation={} global_best={:.6g} evaluations={} extra_evaluations={}".format(
                self.generation,
                self.global_best_value,
                self.total_evaluations,
                self.extra_evaluations,
            )
        )

    def _evaluate(self, positions):
        return np.asarray(self.objective(positions), dtype=np.float64)

    def _apply_boundary(self):
        if self.boundary == "none":
            return
        clipped = np.clip(self.positions, self.lower_bound, self.upper_bound)
        if self.boundary == "clip_zero_velocity":
            hit = clipped != self.positions
            self.velocities[hit] = 0.0
        elif self.boundary != "clip":
            raise ValueError("Unknown PSO boundary mode: {}".format(self.boundary))
        self.positions = clipped

    def _decode_actions(self, actions):
        arr = np.asarray(actions)
        if arr.ndim == 2 and arr.shape[-1] == 3:
            arr = np.argmax(arr, axis=-1)
        elif arr.ndim == 2 and arr.shape[-1] == 1:
            arr = arr[:, 0]
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.argmax(arr, axis=-1)
        arr = arr.astype(np.int64).reshape(self.num_particles)
        if np.any((arr < 0) | (arr > 2)):
            raise ValueError("PSO leader actions must be in {0, 1, 2}.")
        return arr

    def _candidate_indices(self, particle_id):
        return np.array(
            [
                (particle_id - 1) % self.num_particles,
                particle_id,
                (particle_id + 1) % self.num_particles,
            ],
            dtype=np.int64,
        )

    def _leader_indices(self, actions):
        ids = np.arange(self.num_particles)
        left = (ids - 1) % self.num_particles
        right = (ids + 1) % self.num_particles
        leaders = np.where(actions == self.ACTION_LEFT, left, np.where(actions == self.ACTION_RIGHT, right, ids))
        return leaders.astype(np.int64)

    def _get_obs(self):
        span = max(self.upper_bound - self.lower_bound, 1e-12)
        dist_norm = span * np.sqrt(self.dimension)
        obs = np.zeros((self.num_particles, self.obs_dim), dtype=np.float32)
        max_adoptions = max(1.0, float(max(1, self.generation) * self.num_particles))
        for i in range(self.num_particles):
            speed = np.linalg.norm(self.velocities[i]) / dist_norm
            self_dist = np.linalg.norm(self.positions[i] - self.pbest_positions[i]) / dist_norm
            own = [
                self.stagnation[i] / max(1.0, self.max_generations),
                speed,
                self_dist,
                self.last_improvements[i] / self.reward_scale,
            ]
            cand_features = []
            for j in self._candidate_indices(i):
                relative_fit = (self.pbest_values[i] - self.pbest_values[j]) / self.reward_scale
                distance = np.linalg.norm(self.pbest_positions[i] - self.pbest_positions[j]) / dist_norm
                age = self.pbest_age[j] / max(1.0, self.max_generations)
                adopted = self.adopted_counts[j] / max_adoptions
                cand_features.extend([relative_fit, distance, age, adopted])
            remaining = 1.0 - self.generation / max(1.0, self.max_generations)
            obs[i] = np.asarray(own + cand_features + [remaining], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _agent_info(self, particle_id, action, leader_index):
        return {
            "generation": self.generation,
            "particle": particle_id,
            "action": int(action),
            "leader": int(leader_index),
            "leader_memory_version": int(self.last_leader_versions[particle_id]),
            "global_best": float(self.global_best_value),
            "global_best_index": int(self.global_best_index),
            "reward_scale": float(self.reward_scale),
            "evaluations": int(self.total_evaluations),
            "extra_evaluations": int(self.extra_evaluations),
        }
