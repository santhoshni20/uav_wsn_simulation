import numpy as np

from config import *
from nodes import SensorNode, UAV, CommandStation


class SimulationEnvironment:

    def __init__(self):

        self.width = AREA_WIDTH
        self.height = AREA_HEIGHT

        self.current_step = 0

        self.uav = UAV(
            UAV_INITIAL_POSITION,
            UAV_INITIAL_ENERGY
        )

        self.command_station = CommandStation(
            COMMAND_STATION_POSITION
        )

        self.sensors = []

        self.create_sensors()


    def create_sensors(self):

        for i in range(NUM_SENSORS):

            position = (
                np.random.uniform(
                    CORRIDOR_X_MIN,
                    CORRIDOR_X_MAX
                ),
                np.random.uniform(
                    CORRIDOR_Y_MIN,
                    CORRIDOR_Y_MAX
                )
            )

            velocity = np.random.uniform(
                SENSOR_MIN_SPEED,
                SENSOR_MAX_SPEED
            )

            direction = np.random.uniform(
                0,
                2 * np.pi
            )

            energy = np.random.uniform(
                INITIAL_SENSOR_ENERGY_MIN,
                INITIAL_SENSOR_ENERGY_MAX
            )

            priority = np.random.randint(
                MIN_PRIORITY,
                MAX_PRIORITY + 1
            )

            sensor = SensorNode(
                i,
                position,
                velocity,
                direction,
                energy,
                priority
            )

            self.sensors.append(sensor)


    def move_sensors(self):

        for sensor in self.sensors:

            x, y = sensor.position

            new_x = (
                x
                + sensor.velocity
                * np.cos(sensor.direction)
                * TIME_STEP
            )

            new_y = (
                y
                + sensor.velocity
                * np.sin(sensor.direction)
                * TIME_STEP
            )

            # Keep sensors inside the simulation area

            if new_x <= 0 or new_x >= self.width:

                sensor.direction = np.pi - sensor.direction

                new_x = np.clip(
                    new_x,
                    0,
                    self.width
                )

            if new_y <= 0 or new_y >= self.height:

                sensor.direction = -sensor.direction

                new_y = np.clip(
                    new_y,
                    0,
                    self.height
                )

            sensor.position = (
                new_x,
                new_y
            )

            # Energy consumption due to movement

            sensor.energy -= 0.01

            sensor.energy = max(
                sensor.energy,
                0
            )


    def step(self):

        self.move_sensors()

        self.current_step += 1

        return self.get_state()


    def get_state(self):

        return {
            "sensors": self.sensors,
            "uav": self.uav,
            "command_station": self.command_station,
            "step": self.current_step
        }


    def reset(self):

        self.current_step = 0

        self.sensors = []

        self.uav = UAV(
            UAV_INITIAL_POSITION,
            UAV_INITIAL_ENERGY
        )

        self.create_sensors()

        return self.get_state()