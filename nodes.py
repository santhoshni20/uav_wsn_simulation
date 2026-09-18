class SensorNode:

    def __init__(
        self,
        node_id,
        position,
        velocity,
        direction,
        energy,
        priority
    ):

        self.node_id = node_id
        self.position = position
        self.velocity = velocity
        self.direction = direction
        self.energy = energy
        self.priority = priority


class UAV:

    def __init__(self, position, energy):

        self.position = position
        self.energy = energy


class CommandStation:

    def __init__(self, position):

        self.position = position