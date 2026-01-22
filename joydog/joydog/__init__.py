import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from unitree_api.msg import Request

import json


ROBOT_SPORT_API_ID_DAMP              = 1001
ROBOT_SPORT_API_ID_BALANCESTAND      = 1002
ROBOT_SPORT_API_ID_STOPMOVE          = 1003
ROBOT_SPORT_API_ID_STANDUP           = 1004
ROBOT_SPORT_API_ID_STANDDOWN         = 1005
ROBOT_SPORT_API_ID_RECOVERYSTAND     = 1006
ROBOT_SPORT_API_ID_MOVE              = 1008
ROBOT_SPORT_API_ID_SWITCHGAIT        = 1011
ROBOT_SPORT_API_ID_BODYHEIGHT        = 1013
ROBOT_SPORT_API_ID_SPEEDLEVEL        = 1015
ROBOT_SPORT_API_ID_TRAJECTORYFOLLOW  = 1018
ROBOT_SPORT_API_ID_CONTINUOUSGAIT    = 1019
ROBOT_SPORT_API_ID_MOVETOPOS         = 1036
ROBOT_SPORT_API_ID_SWITCHMOVEMODE    = 1038
ROBOT_SPORT_API_ID_VISIONWALK        = 1101
ROBOT_SPORT_API_ID_HANDSTAND         = 1039
ROBOT_SPORT_API_ID_AUTORECOVERY_SET  = 1040
ROBOT_SPORT_API_ID_FREEWALK          = 1045
ROBOT_SPORT_API_ID_CLASSICWALK       = 1049
ROBOT_SPORT_API_ID_FASTWALK          = 1050
ROBOT_SPORT_API_ID_FREEEULER         = 1051


class JoyDog(Node):
    """A ROS2 Node that prints to the console periodically."""

    def __init__(self):
        super().__init__('joy_node')
        self.sm_pub = self.create_publisher(Request, '/api/sport/request', 10)
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_cb,
            10
        )

    def joy_cb(self, msg: Joy):
        """Method that is periodically called by the timer."""
        if msg.buttons[0] == 1: # X button
            req = Request()
            req.header.identity.api_id = ROBOT_SPORT_API_ID_STANDDOWN
            self.sm_pub.publish(req)
            return
        elif msg.buttons[3] == 1: # Triangle up button
            req = Request()
            req.header.identity.api_id = ROBOT_SPORT_API_ID_STANDUP
            self.sm_pub.publish(req)
            return

        req_dict = dict(
            x=msg.axes[1],
            y=msg.axes[0],
            z=msg.axes[2],
        )
        req = Request()
        req.parameter = json.dumps(req_dict)
        req.header.identity.api_id = ROBOT_SPORT_API_ID_MOVE
        self.sm_pub.publish(req)
        #print(req)
        #self.get_logger().info(str(msg))


def main(args=None):
    """
    The main function.
    :param args: Not used directly by the user, but used by ROS2 to configure
    certain aspects of the Node.
    """
    try:
        rclpy.init(args=args)
        n = JoyDog()
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()