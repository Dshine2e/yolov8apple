# Capture stamp and optical frame; ROI is exclusive bottom/right in original RGB pixels.
std_msgs/Header header
string apple_id
string camera_id
string base_frame
string view_type
float32 confidence
int32 class_id
float32[4] bbox_xyxy
sensor_msgs/RegionOfInterest roi
uint32 image_width
uint32 image_height
# Both images have the capture header, identical ROI, and apple_id in the enclosing message.
sensor_msgs/Image rgb_crop
sensor_msgs/Image mask_crop
# Invalid ROS metric values are NaN, never valid origin. JSON exports use null.
geometry_msgs/PointStamped center_camera
geometry_msgs/PointStamped center_base
geometry_msgs/PointStamped surface_camera
geometry_msgs/PointStamped surface_base
bool center_camera_valid
bool center_base_valid
bool surface_valid
bool surface_base_valid
float64 radius_m
float64 diameter_m
bool radius_valid
float64 radius_px
float64 circularity_top
bool circularity_valid
geometry_msgs/PoseStamped camera_pose_base
bool camera_pose_valid
# Unit vector from apple center toward camera, expressed in base_frame.
geometry_msgs/Vector3Stamped direction_base
bool direction_valid
string measurement_method
string tracking_method
float64 valid_depth_ratio
uint32 valid_depth_points
float64 fit_rmse_m
float64 fit_condition
float64 inlier_ratio
float64 cap_fraction
string occlusion_state
string[] reasons
# Camera K/P/D, image state, units and detailed measurement diagnostics; RFC8259 nulls.
string metadata_json
