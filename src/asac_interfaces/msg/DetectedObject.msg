# bbox_xyxy: 원본 RGB 영상의 [xmin, ymin, xmax, ymax], pixel, 우/하단 제외
int32 class_id
string class_name
float32 confidence
float32[4] bbox_xyxy
# depth_valid=false이면 position, depth_m, range_m은 NaN
bool depth_valid
uint32 valid_depth_samples
# Array.header.frame_id의 optical 좌표계: x 오른쪽, y 아래, z 전방 [m]
geometry_msgs/Point position
# depth_m = 광축 방향 Z, range_m = sqrt(X^2 + Y^2 + Z^2) [m]
float32 depth_m
float32 range_m
