
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <sstream>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
  // loop over all Lidar points and associate them to a 2D bounding box
  cv::Mat X(4, 1, cv::DataType<double>::type);
  cv::Mat Y(3, 1, cv::DataType<double>::type);

  for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
  {
    // assemble vector for matrix-vector-multiplication
    X.at<double>(0, 0) = it1->x;
    X.at<double>(1, 0) = it1->y;
    X.at<double>(2, 0) = it1->z;
    X.at<double>(3, 0) = 1;

    // project Lidar point into camera
    Y = P_rect_xx * R_rect_xx * RT * X;
    cv::Point pt;
    pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
    pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

    vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
    {
      // shrink current bounding box slightly to avoid having too many outlier points around the edges
      cv::Rect smallerBox;
      smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
      smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
      smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
      smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

      // check wether point is within current bounding box
      if (smallerBox.contains(pt))
      {
        enclosingBoxes.push_back(it2);
      }
    } // eof loop over all bounding boxes

    // check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1)
    { 
      // add Lidar point to bounding box
      enclosingBoxes[0]->lidarPoints.push_back(*it1);
    }
  } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
  // create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
  {
    // create randomized color for current 3D object
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top=1e8, left=1e8, bottom=0.0, right=0.0; 
    float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
    {
      // world coordinates
      float xw = (*it2).x; // world position in m with x facing forward from sensor
      float yw = (*it2).y; // world position in m with y facing left from sensor
      xwmin = xwmin<xw ? xwmin : xw;
      ywmin = ywmin<yw ? ywmin : yw;
      ywmax = ywmax>yw ? ywmax : yw;

      // top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // find enclosing rectangle
      top = top<y ? top : y;
      left = left<x ? left : x;
      bottom = bottom>y ? bottom : y;
      right = right>x ? right : x;

      // draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
    }

    // draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
    putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
  }

  // plot distance markers
  float lineSpacing = 2.0; // gap between distance markers
  int nMarkers = floor(worldSize.height / lineSpacing);
  for (size_t i = 0; i < nMarkers; ++i)
  {
    int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
    cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
  }

  // display image
  string windowName = "3D Objects";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, topviewImg);

  if(bWait)
  {
    cv::waitKey(0); // wait for key to be pressed
  }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
  for(cv::DMatch d : kptMatches) {
    auto currkp = kptsCurr[d.trainIdx];
    auto prevkp = kptsCurr[d.queryIdx];
    bool firstpt = boundingBox.roi.contains(currkp.pt);
    if(firstpt) {
      float distance = cv::norm(currkp.pt-prevkp.pt);
      if(distance < 60) {
        boundingBox.keypoints.push_back(currkp);
        boundingBox.kptMatches.push_back(d);
      }
    }
  }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
  // compute distance ratios between all matched keypoints
  vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
  for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
  { // outer kpt. loop 
    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
    { // inner kpt.-loop
      double minDist = 100.0; // min. required distance

      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

      //cout<<distCurr<< " "<< distPrev<<endl;
      if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
      { // avoid division by zero

        double distRatio = distCurr / distPrev;
        distRatios.push_back(distRatio);
      }
    } // eof inner loop over all matched kpts
  }     // eof outer loop over all matched kpts

  // only continue if list of distance ratios is not empty
  if (distRatios.size() == 0)
  {
    TTC = NAN;
    return;
  }

  // STUDENT TASK (replacement for meanDistRatio)
  std::sort(distRatios.begin(), distRatios.end());
  long medIndex = floor(distRatios.size() / 2.0);
  double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

  double dT = 1.0 / frameRate;
  TTC = -dT / (1 - medDistRatio);
  // EOF STUDENT TASK
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
  // auxiliary variables
  double dT = 0.1; // time between two measurements in seconds

  // find closest distance to Lidar points 
  double minXPrev = 1e9, minXCurr = 1e9;
  float boundStart = -0.325;
  float boundEnd = 0.075;
  for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) {
    if (it->y > boundStart and it->y < boundEnd) {
      minXPrev = minXPrev>it->x ? it->x : minXPrev;
    }
  }
  for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) {
    if (it->y > boundStart and it->y < boundEnd) {
      minXCurr = minXCurr>it->x ? it->x : minXCurr;
    }
  }

  cout<<minXPrev<< " " <<minXCurr<<endl;
  // compute TTC from both measurements
  TTC = minXCurr * dT / (minXPrev-minXCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
  //filter keypoints
  std::vector<BoundingBox> bbprev = prevFrame.boundingBoxes;
  std::vector<BoundingBox> bbcurr = currFrame.boundingBoxes;
  std::vector<cv::KeyPoint> prevkp = prevFrame.keypoints;
  std::vector<cv::KeyPoint> currkp = currFrame.keypoints;
  std::map<std::string, int> counter;
  for(cv::DMatch d : matches) {
    auto prevk1 = prevkp[d.queryIdx];
    auto currk2 = currkp[d.trainIdx];

    for(BoundingBox bb1 : bbprev) {
      bool firstpt = bb1.roi.contains(prevk1.pt);
      if(firstpt) {
        for(BoundingBox bb2 : bbcurr) {
          bool secondpt = bb2.roi.contains(currk2.pt);
          if(secondpt) {
            std::string s1 = std::to_string(bb1.boxID);
            std::string s2 = std::to_string(bb2.boxID);
            std::string s3 = s1 + "," + s2;
            counter[s3]++;
          }
        }
      }
    }
  }
  for(auto it = counter.begin(); it != counter.end(); it++) {
    //std::cout<<it->first<< " "<<it->second<<std::endl;
    if(it->second > 50) {
      std::stringstream ss(it->first);
      std::string s1;
      std::string s2;
      std::getline(ss, s1, ',');
      std::getline(ss, s2, ',');
      bbBestMatches[std::stoi(s1)] = std::stoi(s2);
    }
  }
  
  // //new
  // vector<string> classes;
  // ifstream ifs("../dat/yolo/coco.names");
  // string line;
  // while (getline(ifs, line)) classes.push_back(line);
  
  // cv::Mat visImg = currFrame.cameraImg.clone();
  // cv::Mat visImg2 = currFrame.cameraImg.clone();

  // cv::Scalar colors [8] = {
  //   cv::Scalar(0, 255, 0),
  //   cv::Scalar(0, 0, 255),
  //   cv::Scalar(0, 0, 102),
  //   cv::Scalar(0, 204, 204),
  //   cv::Scalar(51, 51, 255),
  //   cv::Scalar(255, 51, 153),
  //   cv::Scalar(153, 0, 76),
  //   cv::Scalar(153, 200, 76)
  // };
  // int colorCount = 0;
  // int mycount = 0;
  // for(auto it = bbBestMatches.begin(); it != bbBestMatches.end(); it++) {
  //   int top, left, width, height;
  //   auto currbox = currFrame.boundingBoxes[it->second];
  //   top = currbox.roi.y;
  //   left = currbox.roi.x;
  //   width = currbox.roi.width;
  //   height = currbox.roi.height;
  //   if(top < 0 || left < 0 || width < 0 || height < 0) {
  //     continue;
  //   }
    
  //   cv::rectangle(visImg, cv::Point(left, top), cv::Point(left+width, top+height),colors[colorCount], 2);
    
  //   string label = cv::format("%i", currbox.boxID);
  //   int baseLine;
  //   cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
  //   top = max(top, labelSize.height);
  //   rectangle(visImg, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
  //   cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);

  //   auto prevbox = prevFrame.boundingBoxes[it->first];
  //   top = prevbox.roi.y;
  //   left = prevbox.roi.x;
  //   width = prevbox.roi.width;
  //   height = prevbox.roi.height;
  //   cv::rectangle(visImg2, cv::Point(left, top), cv::Point(left+width, top+height),colors[colorCount], 2);

  //   label = cv::format("%i", prevbox.boxID);
  //   labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
  //   top = max(top, labelSize.height);
  //   rectangle(visImg2, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
  //   cv::putText(visImg2, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);

  //   colorCount++;
  //   if(colorCount >=7) {
  //     colorCount = 0;
  //   }
  // }

  // string windowName = "curr";
  // cv::namedWindow( windowName, 1 );
  // cv::imshow( windowName, visImg );
  // windowName = "prev";
  // cv::namedWindow( windowName, 1 );
  // cv::imshow( windowName, visImg2 );
  // cv::waitKey(0); // wait for key to be pressed
  // //new
}
