#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

using namespace cv;
using namespace cv::dnn;

#define DETECT_INTERVAL 10 // Cứ 10 frame thì chạy detect lại
#define IOU_THRESHOLD 0.4  // Ngưỡng IoU để gán detection cho tracker
#define EMO_W 64           // Chiều rộng input cho MobileNet
#define EMO_H 64           // Chiều cao input cho MobileNet
const std::string LOG_DIR = "face_logs"; // Thư mục gốc để lưu trữ

double calculateIoU(const Rect2d& rect1, const Rect2d& rect2) {
    Rect intersection = rect1 & rect2;
    double area_i = intersection.area();
    double area_u = rect1.area() + rect2.area() - area_i;
    if (area_u > 0) return area_i / area_u;
    return 0.0;
}

int main() {
    // 0. Khởi tạo thư mục gốc
    if (std::filesystem::create_directories(LOG_DIR)) std::cout << "Create log folder: " << LOG_DIR << "\n";
    else std::cout << "Log folder existed, overide old log folder: " << LOG_DIR << std::endl;

    // 1. Cấu hình YuNet
    const int inputW = 640;
    const int inputH = 480;
    Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
        "face_detection_yunet_2023mar_int8.onnx", "",
        Size(inputW, inputH), 0.78f, 0.3f, 50
        // 0.78f: ngưỡng tin cậy, chỉ lấy những dự đoán có điểm tin cậy > ngưỡng
        // 0.3f: ngưỡng IoU dùng trong lọc NMS
        // 50: chỉ lấy tối đa 50 dự đoán sau 2 bước lọc trên
    );
    detector->setInputSize(Size(inputW, inputH)); // Có thể bỏ
    if (!detector) {
        std::cerr << "Can't create FaceDetectorYN. Check ONNX model directory." << std::endl;
        return -1;
    }

    // 2. Cấu hình MobileNet
    Net emotionNet = readNetFromONNX("MobileNet_custom.onnx");
    if (emotionNet.empty()) {
        std::cerr << "Can't load MobileNet model emotion_regresstion.onnx." << std::endl;
        return -1;
    }
    std::vector<std::string> emotion_labels = { "Happy", "Sad", "Surprise", "Angry", "Disgust" };
    int num_frame = emotion_labels.size();

    // 3. Mở webcam, thiết lập thông số Tracking
    VideoCapture cap(0); // 0: webcam. Có thể truyền vào đường dẫn video, ví dụ: "video.mp4"
    if (!cap.isOpened()) {
        std::cerr << "Can't open webcam or file video!" << std::endl;
        return -1;
    }
    Mat frame, img;
    int frame_count = 0;
    long next_face_id = 0;
    std::map<long, std::pair<Ptr<Tracker>, Rect2d>> active_trackers;

    std::map<long, std::ofstream> emotion_logs; // Map để giữ các luồng file CSV đang mở
    // Vector 2D để lưu cảm xúc trung bình của mỗi người (mỗi ID), định dạng: [[Happy, ..., Disgust, Num_frame],...]
    std::vector<std::vector<float>> average;
    std::cout << "Start YuNet Detection and Tracking. Press ESC to exit." << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        // Resize frame để khớp với kích thước input của YuNet
        resize(frame, frame, Size(inputW, inputH), 0, 0, INTER_AREA);

        // 4. Phát hiện khuôn mặt bằng YuNet định kỳ
        if (frame_count % DETECT_INTERVAL == 0 || active_trackers.empty()) {
            Mat faces;
            detector->detect(frame, faces);
            std::vector<Rect2d> current_detections;
            if (!faces.empty()) {
                for (int i = 0; i < faces.rows; i++) {
                    float x = faces.at<float>(i, 0);
                    float y = faces.at<float>(i, 1);
                    float w = faces.at<float>(i, 2);
                    float h = faces.at<float>(i, 3);
                    current_detections.emplace_back(x, y, w, h);
                }
            }

            // A. Xóa các tracker cũ không trùng với phát hiện mới (Giữ nguyên)
            std::vector<long> trackers_to_remove;
            for (auto const& pair_item : active_trackers) {
                long id = pair_item.first;
                const auto& tracker_data = pair_item.second;
                bool found_overlap = false;
                for (const auto& det_rect : current_detections) {
                    if (calculateIoU(tracker_data.second, det_rect) > IOU_THRESHOLD) {
                        found_overlap = true;
                        break;
                    }
                }
                if (!found_overlap) {
                    trackers_to_remove.push_back(id);
                }
            }
            for (long id : trackers_to_remove) {
                // Đóng luồng file khi tracker bị xóa
                if (emotion_logs.count(id)) {
                    emotion_logs[id].close();
                    emotion_logs.erase(id);
                }
                active_trackers.erase(id);
            }

            // B. Cập nhật hoặc thêm mới các tracker
            for (const auto& new_face_box : current_detections) {
                bool found_existing_tracker = false;
                for (auto& pair_item : active_trackers) {
                    auto& tracker_data = pair_item.second;

                    if (calculateIoU(tracker_data.second, new_face_box) > IOU_THRESHOLD) {
                        tracker_data.second = new_face_box;
                        tracker_data.first->init(frame, tracker_data.second);
                        found_existing_tracker = true;
                        break;
                    }
                }

                if (!found_existing_tracker) {
                    Ptr<Tracker> new_tracker = TrackerNano::create();
                    if (new_tracker) {
                        new_tracker->init(frame, new_face_box);

                        // Xử lý lưu trữ cho ID mới
                        long new_id = next_face_id;
                        std::string id_str = "ID" + std::to_string(new_id);
                        std::string face_dir_path = LOG_DIR + "/" + id_str;

                        // 1. Tạo thư mục ID
                        if (std::filesystem::create_directories(face_dir_path)) {
                            std::cout << "Created folder for " << id_str << std::endl;
                        }

                        // 2. Lưu frame đầu tiên với bounding box
                        Mat first_frame_copy = frame.clone();
                        // Vẽ box và ID lên frame copy
                        rectangle(first_frame_copy, new_face_box, Scalar(0, 0, 255), 2);
                        putText(first_frame_copy, id_str, Point(new_face_box.x, new_face_box.y-10),
                            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                        imwrite(face_dir_path + "/first_frame.jpg", first_frame_copy); // Lưu file ảnh

                        // 3. Tạo và mở file CSV
                        std::ofstream csv_file(face_dir_path + "/emotions.csv");
                        if (csv_file.is_open()) {
                            csv_file << "frame,Happy,Sad,Surprise,Angry,Disgust\n"; // Ghi header
                            emotion_logs[new_id] = std::move(csv_file); // Lưu luồng file vào map
                        }
                        else std::cerr << "Can't create file CSV for " << id_str << std::endl;

                        active_trackers[next_face_id++] = { new_tracker, new_face_box };
                    }
                    else std::cerr << "Can't create new Tracker Nano!" << std::endl;
                }
            }
        }

        // 5. Cập nhật tất cả các tracker đang hoạt động
        std::vector<long> failed_trackers;
        for (auto& pair_item : active_trackers) {
            long id = pair_item.first;
            auto& tracker_data = pair_item.second;
            Rect current_roi = tracker_data.second;
            bool ok = tracker_data.first->update(frame, current_roi);

            if (ok) {
                tracker_data.second = current_roi;
                rectangle(frame, current_roi, Scalar(0, 255, 0), 2, 1);
                putText(frame, "ID: " + std::to_string(id), Point(current_roi.x, current_roi.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);

                // Đảm bảo current_roi nằm trong biên ảnh
                current_roi.x = std::max(0, current_roi.x);
                current_roi.y = std::max(0, current_roi.y);
                current_roi.width = std::min(current_roi.width, frame.cols - current_roi.x);
                current_roi.height = std::min(current_roi.height, frame.rows - current_roi.y);

                // Kiểm tra tránh ROI rỗng
                if (current_roi.width <= 0 || current_roi.height <= 0) continue;

                // 6. Phân loại cảm xúc bằng MobileNet
                Mat face = frame(current_roi);
                Mat resized;
                resize(face, resized, Size(EMO_W, EMO_H));
                Mat blob = dnn::blobFromImage(resized, 1.0/255.0, Size(EMO_W, EMO_H), Scalar(), true);
                emotionNet.setInput(blob);
                Mat prob = emotionNet.forward();

                Point classIdPoint;
                double confidence;
                minMaxLoc(prob, nullptr, &confidence, nullptr, &classIdPoint);
                int label_id = classIdPoint.x;

                // Ghi thông tin frame và emotion vào file csv
                emotion_logs.at(id) << frame_count;
                for (int i = 0; i < prob.cols; i++) {
                    float val = prob.at<float>(0, i);
                    emotion_logs.at(id) << "," << std::fixed << std::setprecision(2) << val;
                }
                emotion_logs.at(id) << "\n";

                if (average.size() < id+1) { // Nếu chưa có ID này thì tạo mới
                    average.push_back(prob);
                    average[id].push_back(1);
                }
                else { // Nếu đã có ID này thì cập nhật cảm xúc trung bình
                    for (int i = 0; i < num_frame; i++) {
                        average[id][i] = (average[id][i] * average[id][num_frame] + prob.at<float>(0, i)) / (average[id][num_frame] + 1);
                    }
                    average[id][num_frame] += 1;
                }

                // --- Hiển thị ---
                std::string label = emotion_labels[label_id] + format(" (%.2f)", confidence);
                putText(frame, label, Point(current_roi.x, current_roi.y + 5),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            }
            else failed_trackers.push_back(id);
        }

        for (long id : failed_trackers) {
            // Đóng luồng file khi tracker bị xóa
            if (emotion_logs.count(id)) {
                emotion_logs[id].close();
                emotion_logs.erase(id);
            }
            active_trackers.erase(id);
        }

        putText(frame, "Frame: " + std::to_string(frame_count) + " | Trackers: " + std::to_string(active_trackers.size()),
            Point(10, frame.rows - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        imshow("YuNet Detection + Tracking Nano + MobileNet classification", frame);
        if (waitKey(1) == 27) break;
        frame_count++;
    }

    // Đóng tất cả các luồng file đang mở
    for (auto& pair_item : emotion_logs) {
        if (pair_item.second.is_open()) {
            pair_item.second.close();
        }
    }
    std::cout << "Closed all log file." << std::endl;

    // Ghi cảm xúc trung bình của mỗi người ra màn hình
    for (int i = 0; i < average.size(); i++) {
        for (int j = 0; j < average[i].size(); j++) {
            std::cout << average[i][j] << " ";
        }
        std::cout << "\n";
    }
    cap.release();
    destroyAllWindows();

    // Ghi file csv lưu kết quả tổng (cảm xúc trung bình)
    std::string filename = LOG_DIR + "/" + "output.csv";
    std::ofstream csv_file(filename);
    if (csv_file.is_open()) csv_file << "ID,Happy,Sad,Surprise,Angry,Disgust,Num_frame\n"; // Ghi header
    else std::cerr << "Can't create file CSV for " << filename << "\n";

    int id_temp = 0;
    for (const auto& row : average) {
        csv_file << id_temp << ",";
        for (int i = 0; i < row.size(); i++) {
            csv_file << row[i];
            if (i != row.size() - 1)
                csv_file << ",";
        }
        csv_file << "\n";
        id_temp++;
    }
    
    csv_file.close();
    std::cout << "Writed data to " << filename << " completed." << std::endl;
    return 0;
}