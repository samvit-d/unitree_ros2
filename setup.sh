#!/bin/bash
echo "Setup unitree ros2 environment"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="enp2s0" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
