import math
import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper, ManualControlViewer
from typing import List, Type, Optional, Dict, Any
from DumbMutationModel import DumbMutationModel
import carla
import numpy as np
import random
import gymnasium as gym
import asyncio

class RoarCompetitionRule:
    def __init__(
        self,
        waypoints : List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_carla.RoarPyCarlaActor,
        world: roar_py_carla.RoarPyCarlaWorld
    ) -> None:
        self.waypoints = waypoints
        # self.waypoint_occupancy = np.zeros(len(waypoints),dtype=np.bool_)
        self.vehicle = vehicle
        self.world = world
        self._last_vehicle_location = vehicle.get_3d_location()
        self._respawn_location = None
        self._respawn_rpy = None

    def initialize_race(self):
        self._last_vehicle_location = self.vehicle.get_3d_location()
        vehicle_location = self._last_vehicle_location
        closest_waypoint_dist = np.inf
        closest_waypoint_idx = 0
        for i,waypoint in enumerate(self.waypoints):
            waypoint_dist = np.linalg.norm(vehicle_location - waypoint.location)
            if waypoint_dist < closest_waypoint_dist:
                closest_waypoint_dist = waypoint_dist
                closest_waypoint_idx = i
        self.waypoints = self.waypoints[closest_waypoint_idx+1:] + self.waypoints[:closest_waypoint_idx+1]
        self.furthest_waypoints_index = 0
        print(f"total length: {len(self.waypoints)}")
        self._respawn_location = self._last_vehicle_location.copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()
        # print(self.waypoints[1200:1210])


    def lap_finished(
        self, 
        check_step = 5
    ):
        # print(len(self.waypoints))
        return self.furthest_waypoints_index + check_step >= len(self.waypoints), self.furthest_waypoints_index
        #return np.all(self.waypoint_occupancy)

    async def tick(
        self, 
        check_step = 15
    ):
        current_location = self.vehicle.get_3d_location()
        #print(f"current location at : {current_location}")
        delta_vector = current_location - self._last_vehicle_location
        delta_vector_norm = np.linalg.norm(delta_vector)
        delta_vector_unit = (delta_vector / delta_vector_norm) if delta_vector_norm >= 1e-5 else np.zeros(3)

        previous_furthest_index = self.furthest_waypoints_index
        min_dis = np.inf
        min_index = 0
        #print(f"Previous furthest index {previous_furthest_index}")
        endind_index = previous_furthest_index + check_step if (previous_furthest_index + check_step <= len(self.waypoints)) else len(self.waypoints)
        for i,waypoint in enumerate(self.waypoints[previous_furthest_index:endind_index]):
            waypoint_delta = waypoint.location - current_location
            projection = np.dot(waypoint_delta,delta_vector_unit)
            projection = np.clip(projection,0,delta_vector_norm)
            closest_point_on_segment = current_location + projection * delta_vector_unit
            distance = np.linalg.norm(waypoint.location - closest_point_on_segment)
            #print(f"looking forward index {i}, distance {distance}")
            if distance < min_dis:
                min_dis = distance
                min_index = i
        
        self.furthest_waypoints_index += min_index #= new_furthest_index
        self._last_vehicle_location = current_location
        #print(f"reach waypoints {self.furthest_waypoints_index} at {self.waypoints[self.furthest_waypoints_index].location}")
        #print(f"reach waypoints {self.furthest_waypoints_index}")

    
    async def respawn(
        self
    ):
        # vehicle_location = self.vehicle.get_3d_location()
        # 
        # closest_waypoint_dist = np.inf
        # closest_waypoint_idx = 0
        # for i,waypoint in enumerate(self.waypoints):
        #     waypoint_dist = np.linalg.norm(vehicle_location - waypoint.location)
        #     if waypoint_dist < closest_waypoint_dist:
        #         closest_waypoint_dist = waypoint_dist
        #         closest_waypoint_idx = i
        # closest_waypoint = self.waypoints[closest_waypoint_idx]
        # closest_waypoint_location = closest_waypoint.location
        # closest_waypoint_rpy = closest_waypoint.roll_pitch_yaw
        # self.vehicle.set_transform(
        #     closest_waypoint_location + self.vehicle.bounding_box.extent[2] + 0.2, closest_waypoint_rpy
        # )
        self.vehicle.set_transform(
            self._respawn_location, self._respawn_rpy
        )
        self.vehicle.set_linear_3d_velocity(np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        for _ in range(20):
            await self.world.step()
        
        self._last_vehicle_location = self.vehicle.get_3d_location()
        self.furthest_waypoints_index = 0

async def evaluate_solution(
    world : roar_py_carla.RoarPyCarlaWorld,
    solution_constructor : Type[RoarCompetitionSolution],
    max_seconds = 400,
    enable_visualization : bool = False,
    model = None
) -> Optional[Dict[str, Any]]:
    if enable_visualization:
        viewer = ManualControlViewer()
    
    # Spawn vehicle and sensors to receive data
    waypoints = world.maneuverable_waypoints
    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        waypoints[0].location + np.array([0,0,1]),
        waypoints[0].roll_pitch_yaw,
        True,
    )
    assert vehicle is not None
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
        np.array([0, 10/180.0*np.pi, 0]), # relative rotation
        image_width=1024,
        image_height=768
    )
    location_sensor = vehicle.attach_location_in_world_sensor()
    velocity_sensor = vehicle.attach_velocimeter_sensor()
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
    occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
        50,
        50,
        2.0,
        2.0
    )
    collision_sensor = vehicle.attach_collision_sensor(
        np.zeros(3),
        np.zeros(3)
    )

    assert camera is not None
    assert location_sensor is not None
    assert velocity_sensor is not None
    assert rpy_sensor is not None
    assert occupancy_map_sensor is not None
    assert collision_sensor is not None

    # Start to run solution 
    solution : RoarCompetitionSolution = solution_constructor(
        waypoints,
        RoarCompetitionAgentWrapper(vehicle),
        camera,
        location_sensor,
        velocity_sensor,
        rpy_sensor,
        occupancy_map_sensor,
        collision_sensor,
        model
    )
    #rule = RoarCompetitionRule(waypoints * 3,vehicle,world) # 3 laps
    rule = RoarCompetitionRule(waypoints,vehicle,world) # 1 laps

    for _ in range(20):
        await world.step()
    
    rule.initialize_race()
    # vehicle.close()
    # exit()

    # Timer starts here 
    start_time = world.last_tick_elapsed_seconds
    current_time = start_time
    await vehicle.receive_observation()
    await solution.initialize()

    dist = 0
    
    while True:
        # terminate if time out
        current_time = world.last_tick_elapsed_seconds
        if current_time - start_time > max_seconds:
            vehicle.close()
            return {
                "elapsed_time": max_seconds,
                "distance": dist
            }
        
        # receive sensors' data
        await vehicle.receive_observation()

        await rule.tick()

        if current_time - start_time > 20 and dist < 20:
            vehicle.close()
            return {
                "elapsed_time": max_seconds,
                "distance": dist
            }
        # terminate if there is major collision
        collision_impulse_norm = np.linalg.norm(collision_sensor.get_last_observation().impulse_normal)
        if collision_impulse_norm > 100.0:
            # vehicle.close()
            print(f"major collision of tensity {collision_impulse_norm}")
            # return None
            #await rule.respawn()
            break

        rlf, dist = rule.lap_finished()
        if rlf:
            break
        
        if enable_visualization:
            if viewer.render(camera.get_last_observation()) is None:
                vehicle.close()
                return None

        await solution.step()
        await world.step()

    end_time = world.last_tick_elapsed_seconds
    vehicle.close()
    if enable_visualization:
        viewer.close()
    
    return {
        "elapsed_time" : end_time - start_time,
        "distance": dist
    }

async def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)
    best = DumbMutationModel()
    #n = np.load("b.npz")
    #best.w1 = n['arr_0']
    #best.w2 = n['arr_1']
    #best.w3 = n['arr_2']
    #best.b1 = n['arr_3']
    #best.b2 = n['arr_4']
    #best.b3 = n['arr_5']
    gens = 5
    for i in range(gens):
        models = []
        fitness = []
        pop = 100
        for k in range(pop):
            if (i != -1):
                m = DumbMutationModel(False, best.w1, best.w2, best.w3, best.b1, best.b2, best.b3)
                await m.mutate_all_nodes(0.2)
            else:
                m = DumbMutationModel()
            models.append(m)

            result = await evaluate_solution(
                world,
                RoarCompetitionSolution,
                max_seconds=400,
                enable_visualization=False,
                model=m
            )
            print(f"gen {i} member {k} got to {result['distance']} in {result['elapsed_time']}s")
            if (result['elapsed_time'] < 399):
                fitness.append(math.pow((2771 - result["distance"])/2770, 3) * (result["elapsed_time"] ** 2))
                np.savez(f"backups/it_d_gen_{i}_no_{k}_t_{result['elapsed_time']}s_d_{result['distance']}.npz", best.w1, best.w2, best.w3, best.b1, best.b2, best.b3)
            else:
                models.remove(m)
        b = fitness.index(min(fitness))
        print(f"best fitness for this generation: {fitness[b]}")
        best = models[b]
        np.savez(f"modelbest_{i}_b.npz", best.w1, best.w2, best.w3, best.b1, best.b2, best.b3)


if __name__ == "__main__":
    asyncio.run(main())
