#include <sys/time.h>

#include <string.h>
#include <iostream>     
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <android/log.h>
#include <jni.h>

#include <cvd/image.h>
#include <cvd/image_io.h>
#include <cvd/byte.h>
#include <cvd/utility.h>
#include <cvd/convolution.h>
#include <cvd/vision.h>

#include "MahonyAHRS.h"
#include "SmallBlurryImage.h"
#include "ATANCamera.h"

using namespace std;

extern "C"{

struct timeval tstart;
struct timeval tbegin;
struct timeval tend;

static bool isFirstUpdate = true;
static int imuCount = 0;

static bool isFirstImage = true;
CVD::Image<CVD::byte> imageData;
CVD::ImageRef SBISize;
TooN::SO3<> pose_reset;
float state_reset[7];
int frameCount = 0;

SmallBlurryImage SBI;
SmallBlurryImage SI;
int countFromCorrection = 0;

/*
* Code for imu pose.
*/

// Init imu pose
void InitIMU(float* pimuval, float* q)
{
    MahonyAHRS::init(pimuval[0], pimuval[1], pimuval[2], q);
}

// Reset
JNIEXPORT void JNICALL
Java_com_citrus_slam_MahonyAHRS_QuaternionSensor_nativeReset( JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_INFO, "JNIMsg", "JNI nativeReset called");
    MahonyAHRS::reset();
    isFirstUpdate = true;
    isFirstImage = true;
    imuCount = 0;
    frameCount = 0;
}

// Update By IMU
JNIEXPORT void JNICALL
Java_com_citrus_slam_MahonyAHRS_QuaternionSensor_nativeUpdateIMU( JNIEnv* env, jobject thiz,
        jfloatArray imuval, jfloatArray q)
{

    float* pimuval = env->GetFloatArrayElements(imuval, 0);
    float* pq = env->GetFloatArrayElements(q, 0);

    float imufreq = 50.f;      // the frequency is important

    if (isFirstUpdate) {        // if first update imu, init it
        gettimeofday(&tstart, 0);
        gettimeofday(&tbegin, 0);
        isFirstUpdate = false;
        InitIMU(pimuval, pq);
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                "JNI Init IMU called, init q : %f %f %f %f", pq[0], pq[1], pq[2], pq[3]);
    } else {

        gettimeofday(&tend, 0);
        imufreq = 1000000u/ (((tend.tv_sec - tstart.tv_sec)*1000000u
                + tend.tv_usec - tstart.tv_usec));        // caculate the frequency
        gettimeofday(&tstart, 0);
    }

    MahonyAHRS::updateIMU(pimuval[3], pimuval[4], pimuval[5], pimuval[0], pimuval[1], pimuval[2],
            imufreq, pq[0], pq[1], pq[2], pq[3]);

    imuCount++;
    if (imuCount==500) {
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg", "JNI nativeUpdateIMU called,"
                "imufreq: %f, q : %f %f %f %f", imufreq, pq[0], pq[1], pq[2], pq[3]);
        imuCount=0;
    }

    env->ReleaseFloatArrayElements(imuval, pimuval, 0);
    env->ReleaseFloatArrayElements(q, pq, 0);
}

/*
* Following code is for vision correction.
*/

// Reset Vidsion
JNIEXPORT void JNICALL
Java_com_citrus_slam_MahonyAHRS_QuaternionSensor_nativeResetVision( JNIEnv* env, jobject thiz)
{
    isFirstImage = true;
}

TooN::Vector<4> QFromW(const TooN::Vector<3> & w) {
    double rad = sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
    double cos_rad_2 = cos(rad/2);
    double sin_rad_2 = sin(rad/2);

    return TooN::makeVector(cos_rad_2, w[0]/rad*sin_rad_2, w[1]/rad*sin_rad_2, w[2]/rad*sin_rad_2);
}

TooN::Vector<3> WFromQ(const TooN::Vector<4> & q) {
    double rad = acos(q[0])*2;
    double len = sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    return TooN::makeVector(rad*q[1]/len, rad*q[2]/len, rad*q[3]/len );
}

std::pair< TooN::SO3<>, double> CalcSBIRotation()
{
    static ATANCamera mCamera("/sdcard/calibration/calibration.txt");               // to be inited from file
    std::pair<TooN::SE2<>, double> result_pair;
    result_pair = SI.IteratePosRelToTarget(SBI, 10);
    TooN::SE3<> se3Adjust = SmallBlurryImage::SE3fromSE2(result_pair.first, mCamera);
    //__android_log_print(ANDROID_LOG_INFO, "JNIMsg",
    //                   "JNI CalcSBIRotation called,"
    //                    "error: %f",
    //                    result_pair.second);
    return std::pair< TooN::SO3<>, double >(se3Adjust.get_rotation(), result_pair.second);
    //return std::pair< TooN::SO3<>, double >(TooN::SO3<>(), 3000000);
}

TooN::SO3<> loadIMUExtrinsic() {
    SE3<> res;
    ifstream file("/sdcard/calibration/IMUExtrinsic.txt");

    if (!file.is_open()) {
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                           "JNI loadIMUExtrinsic called,"
                            "warning: open IMUExtrinsic.txt failed !");
        return res.get_rotation();
    }
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    ss >> res;

    std::stringstream outlog;
    outlog << res;

    __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                               "JNI loadIMUExtrinsic called,"
                                "SE3: %s", outlog.str().c_str());

    return res.get_rotation();
}

// Update by vision
JNIEXPORT void JNICALL
Java_com_citrus_slam_MahonyAHRS_QuaternionSensor_nativeUpdateVision( JNIEnv* env, jobject thiz,
        jbyteArray imageArray, jint width, jint height)
{

    if (isFirstUpdate) return;      // if has not do imu update,
                                    // return
    static TooN::SO3<> R_ic = loadIMUExtrinsic();

    // get image array data
    int len = env->GetArrayLength(imageArray);
    imageData.resize(CVD::ImageRef(width, height));
    env->GetByteArrayRegion(imageArray, 0, width*height, (jbyte*)imageData.data() );

    static ATANCamera mCamera("/sdcard/calibration/calibration.txt");

    if (isFirstImage) {                     // if first image, setup correction SBI
        SBI = SmallBlurryImage(imageData);
        SBI.MakeJacs();
        MahonyAHRS::getState(state_reset);
        pose_reset = TooN::SO3<>::exp(
                        WFromQ(TooN::makeVector(state_reset[0], state_reset[1],
                            state_reset[2], state_reset[3]))
                        );

        SBISize = SBI.GetSize();
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                "JNI nativeUpdateVision called,"
                "SBISize: %d, %d",
                SBISize.x, SBISize.y);

        isFirstImage = false;
    } else {                                // if new image, do correction routine
        SI = SmallBlurryImage(imageData);
        double diffVal = SI.ZMSSD(SBI);     // compute SSD diff

        if (frameCount == 200) {
            __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                    "JNI nativeUpdateVision called,"
                    "diffVal: %f", diffVal);
            frameCount = 0;
        }

        if (diffVal < 50000) {              // if SSD small , do correction
            float state_now[7];
            MahonyAHRS::getState(state_now);
            std::pair< TooN::SO3<>, double > rotation_pair = CalcSBIRotation();       // rotation from SBI to SI

            if (rotation_pair.second < 15000) {
                TooN::SO3<> pose_correction =
                        pose_reset * R_ic.inverse() * rotation_pair.first.inverse() * R_ic ;        // corrected pose
                TooN::Vector<4> state_correction =
                        QFromW(pose_correction.ln());           // convert to Quaternion

                state_now[0] = state_correction[0];
                state_now[1] = state_correction[1];
                state_now[2] = state_correction[2];
                state_now[3] = state_correction[3];
                state_now[4] = 0;
                state_now[5] = 0;
                state_now[6] = 0;

                MahonyAHRS::setState(state_now);                // set corrected state

                countFromCorrection = 0;
                __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                                "JNI nativeUpdateVision called,"
                                "Reset State, diffVal: %f, error: %f", diffVal, rotation_pair.second);
            }
        }

    }

    frameCount++;
    countFromCorrection++;

    if (countFromCorrection == 400) {               // if too long time from last correction, reset
                                                    // correction
        isFirstImage = true;
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                "JNI nativeUpdateVision,"
                "CountFromCorrection too long,"
                "Reset SBI");
    }
}

}
