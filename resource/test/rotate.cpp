#include <cv2>

int main() {
  cv::Mat input = cv::imread("../inputData/rectangles.png");

  cv::Mat gray;
  cv::cvtColor(input, gray, CV_BGR2GRAY);

  // since your image has compression artifacts, we have to threshold the image
  int threshold = 200;
  cv::Mat mask = gray > threshold;

  cv::imshow("mask", mask);

  // extract contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  for (int i = 0; i < contours.size(); ++i) {
    // fit bounding rectangle around contour
    cv::RotatedRect rotatedRect = cv::minAreaRect(contours[i]);

    // read points and angle
    cv::Point2f rect_points[4];
    rotatedRect.points(rect_points);

    float angle = rotatedRect.angle; // angle

    // read center of rotated rect
    cv::Point2f center = rotatedRect.center; // center

    // draw rotated rect
    for (unsigned int j = 0; j < 4; ++j)
      cv::line(input, rect_points[j], rect_points[(j + 1) % 4],
               cv::Scalar(0, 255, 0));

    // draw center and print text
    std::stringstream ss;
    ss << angle; // convert float to string
    cv::circle(input, center, 5, cv::Scalar(0, 255, 0)); // draw center
    cv::putText(input, ss.str(), center + cv::Point2f(-25, 25),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                cv::Scalar(255, 0, 255)); // print angle
  }
}
