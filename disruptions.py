from config import *


class CommunicationDisruption:

    def __init__(self):

        self.enabled = ENABLE_DISRUPTION


    def is_inside_disruption_zone(self, position):

        x, y = position

        return (
            DISRUPTION_X_MIN <= x <= DISRUPTION_X_MAX
            and
            DISRUPTION_Y_MIN <= y <= DISRUPTION_Y_MAX
        )


    def calculate_link_quality(
        self,
        link_quality,
        position1,
        position2
    ):

        if not self.enabled:
            return link_quality


        midpoint = (
            (position1[0] + position2[0]) / 2,
            (position1[1] + position2[1]) / 2
        )


        if self.is_inside_disruption_zone(midpoint):

            link_quality *= DISRUPTION_LINK_QUALITY_FACTOR


        return link_quality