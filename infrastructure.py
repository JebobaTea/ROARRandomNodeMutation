from roar_py_interface import RoarPyActor, RoarPySensor
import typing
import gymnasium as gym
import roar_py_interface
import pygame
from PIL.Image import Image
import numpy as np
from typing import Optional, Dict, Any

class ManualControlViewer:
    def __init__(
        self
    ):
        self.screen = None
        self.clock = None
        self.last_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0])
        }
    
    def init_pygame(self, x, y) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()

    def close(self) -> None:
        pygame.quit()

    def render(self, image : roar_py_interface.RoarPyCameraSensorData, occupancy_map : Optional[Image] = None) -> Optional[Dict[str, Any]]:
        image_pil : Image = image.get_image()
        occupancy_map_rgb = occupancy_map.convert("RGB") if occupancy_map is not None else None
        if self.screen is None:
            if occupancy_map_rgb is None:
                self.init_pygame(image_pil.width, image_pil.height)
            else:
                self.init_pygame(image_pil.width + occupancy_map.width, image_pil.height)
        
        new_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0])
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_UP]:
            new_control['throttle'] = 0.4
        if pressed_keys[pygame.K_DOWN]:
            new_control['brake'] = 0.2
        if pressed_keys[pygame.K_LEFT]:
            new_control['steer'] = -0.2
        if pressed_keys[pygame.K_RIGHT]:
            new_control['steer'] = 0.2
        
        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        if occupancy_map_rgb is not None:
            occupancy_map_surface = pygame.image.fromstring(occupancy_map_rgb.tobytes(), occupancy_map_rgb.size, occupancy_map_rgb.mode).convert()

        self.screen.fill((0,0,0))
        self.screen.blit(image_surface, (0, 0))
        if occupancy_map_rgb is not None:
            self.screen.blit(occupancy_map_surface, (image_pil.width, 0))

        pygame.display.flip()
        self.clock.tick(60)
        self.last_control = new_control
        return new_control

class RoarCompetitionAgentWrapper(RoarPyActor):
    def __init__(self, wrapped : RoarPyActor):
        self._wrapped = wrapped
    
    @property
    def control_timestep(self) -> float:
        return self._wrapped.control_timestep
    
    @property
    def force_real_control_timestep(self) -> bool:
        return self._wrapped.force_real_control_timestep

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return self._wrapped.get_sensors()

    def get_action_spec(self) -> gym.Space:
        return self._wrapped.get_action_spec()
    
    async def _apply_action(self, action: typing.Any) -> bool:
        return await self._wrapped._apply_action(action)

    def close(self):
        pass

    def is_closed(self) -> bool:
        return self._wrapped.is_closed()

    def __del__(self):
        pass
    
    async def apply_action(self, action: typing.Any) -> bool:
        return await self._wrapped.apply_action(action)

    def get_gym_observation_spec(self) -> gym.Space:
        return self._wrapped.get_gym_observation_spec()

    async def receive_observation(self) -> typing.Dict[str, typing.Any]:
        return await self._wrapped.receive_observation()
    
    def get_last_observation(self) -> typing.Optional[typing.Dict[str,typing.Any]]:
        return self._wrapped.get_last_observation()
    
    def get_last_gym_observation(self) -> typing.Optional[typing.Dict[str,typing.Any]]:
        return self._wrapped.get_last_gym_observation()

    def convert_obs_to_gym_obs(self, observation : typing.Dict[str,typing.Any]) -> typing.Dict[str,typing.Any]:
        return self._wrapped.convert_obs_to_gym_obs(observation)
