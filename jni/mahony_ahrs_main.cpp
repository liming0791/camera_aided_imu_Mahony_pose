#include <sys/time.h>

#include <string.h>
#include <iostream>     
#include <fstream>      
#include <stdlib.h>
#include <android/log.h>
#include <jni.h>

#include <cvd/image.h>
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
struct timeval tend;

static bool isFirstUpdate = true;
static int imuCount = 0;

static bool isFirstImage = true;
CVD::Image<CVD::byte> imageData;
//CVD::Image<float> SBI;
//CVD::Image<float> mimTemplate;
CVD::ImageRef SBISize;
float state_reset[7];
int frameCount = 0;

SmallBlurryImage SBI;
SmallBlurryImage SI;
int countFromCorrection = 0;

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
        isFirstUpdate = false;
        InitIMU(pimuval, pq);
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                "JNI Init IMU called, init q : %f %f %f %f", pq[0], pq[1], pq[2], pq[3]);
    } else {
        // caculate the frequency
        gettimeofday(&tend, 0);
        imufreq = 1 / (((tend.tv_sec - tstart.tv_sec)*1000000u + tend.tv_usec - tstart.tv_usec)/1000000.f);
        gettimeofday(&tstart, 0);
    }

    MahonyAHRS::updateIMU(pimuval[3], pimuval[4], pimuval[5], pimuval[0], pimuval[1], pimuval[2],
            imufreq, pq[0], pq[1], pq[2], pq[3]);

    imuCount++;
    if (imuCount==100) {
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg", "JNI nativeUpdateIMU called,"
                "imufreq: %f, q : %f %f %f %f", imufreq, pq[0], pq[1], pq[2], pq[3]);
        imuCount=0;
    }

    env->ReleaseFloatArrayElements(imuval, pimuval, 0);
    env->ReleaseFloatArrayElements(q, pq, 0);
}

// Reset Vidsion
JNIEXPORT void JNICALL
Java_com_citrus_slam_MahonyAHRS_QuaternionSensor_nativeResetVision( JNIEnv* env, jobject thiz)
{
    isFirstImage = true;
}

double SSD(CVD::Image<float> & sbi1, CVD::Image<float> & sbi2)
{
    double dSSD = 0.0;
    CVD::ImageRef ir;
    do
    {
      double dDiff = sbi1[ir] - sbi2[ir];
      dSSD += dDiff * dDiff;
    }
    while(ir.next(SBISize));
    return dSSD;
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

TooN::SO3<> CalcSBIRotation()
{
    static ATANCamera mCamera("nothing.txt");   // to be inited from file
    std::pair<TooN::SE2<>, double> result_pair;
    result_pair = SI.IteratePosRelToTarget(SBI, 6);
    TooN::SE3<> se3Adjust = SmallBlurryImage::SE3fromSE2(result_pair.first, mCamera);
    return se3Adjust.get_rotation();
}

// Update by vision
JNIEXPORT void JNICALL
Java_com_citrus_slam_MahonyAHRS_QuaternionSensor_nativeUpdateVision( JNIEnv* env, jobject thiz,
        jbyteArray imageArray, jint width, jint height)
{

    if (isFirstUpdate) return;      // if has not do imu update,
                                    // return

    int len = env->GetArrayLength(imageArray);
    imageData.resize(CVD::ImageRef(width, height));
    env->GetByteArrayRegion(imageArray, 0, width*height, (jbyte*)imageData.data() );

/*    if (isFirstImage) {
        while (imageData.size().x > 40) {
            imageData = halfSample(imageData);
        }
        SBISize = imageData.size();
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                "JNI nativeUpdateVision called,"
                "SBISize: %d, %d",
                SBISize.x, SBISize.y);

        // minus mean value
        CVD::ImageRef ir;
        unsigned int nSum = 0;
        do {
            nSum += imageData[ir];
        } while (ir.next(SBISize));

        float fMean = ((float) nSum) / SBISize.area();

        SBI.resize(SBISize);
        ir.home();
        do {
            SBI[ir] = imageData[ir] - fMean;
        } while(ir.next(SBISize));
        // done

        convolveGaussian(SBI, 2.5);
        MahonyAHRS::getState(state_reset);

        isFirstImage = false;

    } else {
        while (imageData.size().x > 40) {
            imageData = halfSample(imageData);
        }
        // minus mean value
        CVD::ImageRef ir;
        unsigned int nSum = 0;
        do {
            nSum += imageData[ir];
        } while (ir.next(SBISize));

        float fMean = ((float) nSum) / SBISize.area();

        mimTemplate.resize(SBISize);
        ir.home();
        do {
            mimTemplate[ir] = imageData[ir] - fMean;
        } while(ir.next(SBISize));
        // done

        convolveGaussian(mimTemplate, 2.5);
        double diffVal = SSD(mimTemplate, SBI);

        if (frameCount == 200) {
            __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                    "JNI nativeUpdateVision called,"
                    "diffVal: %f", diffVal);
            frameCount = 0;
        }

        if (diffVal < 40000) {
            MahonyAHRS::setState(state_reset);
            __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                            "JNI nativeUpdateVision called,"
                            "Reset State, diffVal: %f", diffVal);
        }
    }
*/

    if (isFirstImage) {
        SBI = SmallBlurryImage(imageData);
        SBI.MakeJacs();
        MahonyAHRS::getState(state_reset);

        SBISize = SBI.GetSize();
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                "JNI nativeUpdateVision called,"
                "SBISize: %d, %d",
                SBISize.x, SBISize.y);

        isFirstImage = false;
    } else {
        SI = SmallBlurryImage(imageData);
        double diffVal = SI.ZMSSD(SBI);

        if (frameCount == 200) {
            __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                    "JNI nativeUpdateVision called,"
                    "diffVal: %f", diffVal);
            frameCount = 0;
        }

        if (diffVal < 40000) {
            float state_now[7];
            MahonyAHRS::getState(state_now);
            TooN::SO3<> rotation = CalcSBIRotation();       // rotation from SBI to SI
            TooN::SO3<> pose_reset = TooN::SO3<>::exp(
                    WFromQ(TooN::makeVector(state_reset[0], -state_reset[2],
                        -state_reset[1], -state_reset[3]))
                    );
            TooN::SO3<> pose_correction = pose_reset * rotation.inverse();

            TooN::Vector<4> state_correction = QFromW(pose_correction.ln());

            state_now[0] = state_correction[0];
            state_now[1] = -state_correction[2];
            state_now[2] = -state_correction[1];
            state_now[3] = -state_correction[3];

            MahonyAHRS::setState(state_now);

            countFromCorrection = 0;
            __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                            "JNI nativeUpdateVision called,"
                            "Reset State, diffVal: %f", diffVal);
        }

    }

    frameCount++;
    countFromCorrection++;

    if (frameCount == 400) {
        isFirstImage = true;
        __android_log_print(ANDROID_LOG_INFO, "JNIMsg",
                "JNI nativeUpdateVision,"
                "CountFromCorrection too long,"
                "Reset SBI");
    }
}

}
