float32[] data #the transform [x,y,theta] from ICP
uint16 source_key #the source key ID for the SLAM graph
uint16 target_key #the target key ID for the slam graph
uint8 vin #vehicle ID number that the loop closure was found on
uint8 loop_vin # the vin of the vehcile that this loop closure is with
sensor_msgs/PointCloud2 cloud #the point cloud, this cloud is for vis purposes only