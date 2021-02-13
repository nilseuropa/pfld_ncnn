#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <time.h>

#include "net.h"
#include "cpu.h"
#include "facedetection/facedetectcnn.h"
#include "ncnn_config.h"

#ifdef GPU_SUPPORT
  #include "gpu.h"
  #include "gpu_support.h"
#endif

#define DETECT_BUFFER_SIZE 0x20000

using namespace  cv;

int main(int argc, char** argv)
{
    const char * param_path = "../model/pfld-sim.param";
    const char * bin_path = "../model/pfld-sim.bin";

    ncnn::Net pfld;

    #ifdef GPU_SUPPORT
    g_vkdev = ncnn::get_gpu_device(selectGPU(0));
    g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
    g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    pfld.opt.use_vulkan_compute = true;
    pfld.set_vulkan_device(g_vkdev);
    #endif

    pfld.load_param(param_path);
    pfld.load_model(bin_path);

    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if(!cap.isOpened()){
        std::cout << "video cant be read" << std::endl;
    }

    int capture_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int capture_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frameRate = cap.get(cv::CAP_PROP_FPS);

    cv::Mat frame;
    int * detector_results = NULL;
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not allocate buffer.\n");
        return -1;
    }

    const int num_landmarks = 106 * 2;
    float landmarks[num_landmarks];

    int detector_width = 160;
    int detector_height = 120;

    Mat det_input;

    #ifdef GPU_SUPPORT
      ncnn::create_gpu_instance();
    #endif

    int num_of_threads = ncnn::get_cpu_count();

    while(true){

        TickMeter timer;
        timer.start();
        cap >> frame;
        if(frame.empty()) break;

        resize(frame, det_input, Size(detector_width, detector_height));
        detector_results = facedetect_cnn(pBuffer, (unsigned char*)(det_input.ptr(0)), detector_width, detector_height, (int)det_input.step);
        Mat result_image = frame.clone();

        for(int i = 0; i < (detector_results ? *detector_results : 0); i++)
        {
            short * p = ((short*)(detector_results+1))+142*i;
            int confidence = p[0];
            int x = p[1];
            int y = p[2];
            int w = p[3];
            int h = p[4];
            int shift = w * 0.1;
            x = (x - shift) < 0 ? 0: x - shift;
            y = (y - shift) < 0 ? 0: y - shift;
            w = w + shift * 2;
            h = h + shift * 2;

            x = int(x  * 1.0 / detector_width * capture_width);
            y = int(y  * 1.0 / detector_height * capture_height);
            w = int(w * 1.0 / detector_width * capture_width);
            h = int(h * 1.0 / detector_height * capture_height);
            w = (w > capture_width) ? capture_width : w;
            h = (h > capture_height) ? capture_height : h;

            char sScore[256];
            snprintf(sScore, 256, "%d", confidence);

            if(confidence > 50){

                if(x + w >= capture_width) w = capture_width - x;
                if(y + h >= capture_height) h = capture_height - y;

                cv::putText(result_image, sScore, cv::Point(x, y-8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
                rectangle(result_image, Rect(x, y, w, h), Scalar(0, 0, 255), 1);

                // PFDL
                cv::Mat face_rect = frame(cv::Rect(x, y, w, h));
                cv::resize(face_rect, face_rect, cv::Size(112, 112));

                ncnn::Mat out;
                ncnn::Mat in = ncnn::Mat::from_pixels(face_rect.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);

                const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
                in.substract_mean_normalize(0, norm_vals);
                ncnn::Extractor ex = pfld.create_extractor();

                ex.set_num_threads(8);
                ex.input("input_1", in);
                ex.extract("415", out);

                for (int j = 0; j < out.w; j++)
                {
                    landmarks[j] = out[j];
                }
                for(int i = 0; i < num_landmarks / 2; i++){
                    cv::circle(result_image, cv::Point(landmarks[i * 2] * w + x, landmarks[i * 2 + 1] * h + y),
                               2,cv::Scalar(0, 0, 255), -1);
                }

            }
        }

        timer.stop();
        string fps = "FPS: " + to_string(1000 / timer.getTimeMilli());
        cv::putText(result_image, fps, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        cv::imshow("PFLD", result_image);
        cv::waitKey(1);
    }
    free(pBuffer);
    #ifdef GPU_SUPPORT
      ncnn::destroy_gpu_instance();
    #endif
    return 0;
}
