"""Compatibility shim that maps gymnasium API to classic gym."""
import gym as _gym

Env = _gym.Env
Wrapper = _gym.Wrapper
ObservationWrapper = _gym.ObservationWrapper
RewardWrapper = _gym.RewardWrapper
ActionWrapper = _gym.ActionWrapper
spaces = _gym.spaces
make = _gym.make
