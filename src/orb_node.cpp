#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ORBTrackerNode : public rclcpp::Node
{
public:
  ORBTrackerNode() : Node("orb_tracker_node")
  {
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&ORBTrackerNode::image_callback, this, std::placeholders::_1));

    direction_pub_ = this->create_publisher<std_msgs::msg::String>("/avoid_direction", 10);
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/orb_keypoints_image", 10);

    orb_ = cv::ORB::create();
    bf_ = cv::BFMatcher(cv::NORM_HAMMING);

    RCLCPP_INFO(this->get_logger(), "ORB Tracker Node with direction logic started");
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat gray;
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    // Draw keypoints for visualization
    cv::Mat output_image;
    cv::drawKeypoints(cv_ptr->image, keypoints, output_image);

    // Draw vertical lines to show left, center, right zones
    int width = output_image.cols;
    cv::line(output_image, cv::Point(width / 3, 0), cv::Point(width / 3, output_image.rows), cv::Scalar(255, 0, 0), 2);
    cv::line(output_image, cv::Point(2 * width / 3, 0), cv::Point(2 * width / 3, output_image.rows), cv::Scalar(255, 0, 0), 2);

    // Publish the image with keypoints and grid
    sensor_msgs::msg::Image output_msg;
    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", output_image).toImageMsg(output_msg);
    image_pub_->publish(output_msg);

    if (!prev_keypoints_.empty() && 
        !prev_descriptors_.empty() && 
        !descriptors.empty() && 
        prev_descriptors_.type() == descriptors.type() && 
        prev_descriptors_.cols == descriptors.cols) {

      std::vector<cv::DMatch> matches;
      bf_.match(prev_descriptors_, descriptors, matches);

      float left_growth = 0.0, center_growth = 0.0, right_growth = 0.0;
      int left_count = 0, center_count = 0, right_count = 0;

      for (const auto& match : matches) {
        const auto& kp_prev = prev_keypoints_[match.queryIdx];
        const auto& kp_curr = keypoints[match.trainIdx];

        float growth = kp_curr.size / std::max(kp_prev.size, 1.0f);
        float x = kp_curr.pt.x;

        if (x < width / 3) {
          left_growth += growth;
          left_count++;
        } else if (x < 2 * width / 3) {
          center_growth += growth;
          center_count++;
        } else {
          right_growth += growth;
          right_count++;
        }

        // Draw keypoint center position (for debugging across different cams)
        cv::circle(output_image, kp_curr.pt, 2, cv::Scalar(0, 255, 255), -1);
      }

      float avg_left = left_count > 0 ? left_growth / left_count : 0.0f;
      float avg_center = center_count > 0 ? center_growth / center_count : 0.0f;
      float avg_right = right_count > 0 ? right_growth / right_count : 0.0f;

      std_msgs::msg::String direction_msg;
      float total_expansion = avg_left + avg_center + avg_right;

      // Threshold check for movement — avoid small/no growth scenes
      if (total_expansion > 3.5f) {
        if (avg_left < avg_center && avg_left < avg_right) {
          direction_msg.data = "left";
        } else if (avg_right < avg_left && avg_right < avg_center) {
          direction_msg.data = "right";
        } else {
          direction_msg.data = "back";
        }
      } else {
        direction_msg.data = "forward";
      }

      direction_pub_->publish(direction_msg);
    }

    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr direction_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  cv::Ptr<cv::ORB> orb_;
  cv::BFMatcher bf_;

  std::vector<cv::KeyPoint> prev_keypoints_;
  cv::Mat prev_descriptors_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ORBTrackerNode>());
  rclcpp::shutdown();
  return 0;
}