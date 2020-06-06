/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"
#include "npy.hpp"

constexpr float MIN_DEPTH = 1e-3f;
constexpr float MAX_DEPTH = 800.0f;
// Uses a scale factor of 5.4 due to:
//     A. KITTI real baseline is 0.54m (training set)
//     B. Monodepth2 NN train baseline is 0.1m
constexpr float DEPTH_SCALE_FACTOR = 5.4f;
constexpr int DEFAULT_IMG_WIDTH = 1241;
constexpr int DEFAULT_IMG_HEIGHT = 376;
static int mImageWidth;
static int mImageHeight;
// Experimental correction factor
static float overestimationFactor;

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

void dread(const std::string& imageFilenameD, cv::Mat& imD);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./monod_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Read image size
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    if(fSettings.isOpened()) {
        mImageWidth = fSettings["Camera.width"];
        mImageHeight = fSettings["Camera.height"];
        overestimationFactor = fSettings["OverestimationFactor"];

        if(mImageWidth<1 || mImageHeight<1) {
            mImageWidth = DEFAULT_IMG_WIDTH;
            mImageHeight = DEFAULT_IMG_HEIGHT;
        }

   } else {
        mImageWidth = DEFAULT_IMG_WIDTH;
        mImageHeight = DEFAULT_IMG_HEIGHT;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vstrImageFilenamesD, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONODEPTH,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        dread(vstrImageFilenamesD[ni], imD);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonodepth(im,imD,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);
    vstrImageFilenamesD.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageFilenamesD[i] = strPrefixLeft + ss.str() + "_disp.npy";
    }
}

void cvMatFromNumpy(const std::string& path, cv::Mat& mat) {
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<float> data;

    shape.clear();
    data.clear();
    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    mat.create(shape[2], shape[3], CV_32F);
    float* pMat = mat.ptr<float>();
    for (float d : data) {
        *pMat = d;
        ++pMat;
    }
}

/**
 * Converts disparity (in pixels) to depth (in meters)
 */
void disp_to_depth(const cv::Mat& disp, cv::Mat& depth) {
    depth = 1.0f / disp;
    depth *= DEPTH_SCALE_FACTOR * overestimationFactor;
    cv::threshold(depth, depth, MIN_DEPTH, 0.0, cv::THRESH_TOZERO);
    cv::threshold(depth, depth, MAX_DEPTH, 0.0, cv::THRESH_TRUNC);
}

void dread(const std::string& imageFilenameD, cv::Mat& imD) {
    cvMatFromNumpy(imageFilenameD, imD);
    cv::resize(imD, imD, cv::Size(mImageWidth, mImageHeight));
    disp_to_depth(imD, imD);
}