
//TOUIDJINE Zaki I2A Tutoriel 6 - Suivi AR avec AKAZE et OpenCV

#include <opencv2/opencv.hpp> 
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>

using namespace cv;
using namespace std;

// === Variables globales pour la trackbar ===
int g_trackbar_pos = 0;
int g_total_frames = 1;
bool g_user_is_setting_trackbar = false;

void on_trackbar(int pos, void* userdata) {
    g_trackbar_pos = pos;
    g_user_is_setting_trackbar = true;
}

// === Fonction utilitaire : matches symétriques ===
vector<DMatch> symmetricMatches(const vector<vector<DMatch>>& matches12,
                                const vector<vector<DMatch>>& matches21) {
    vector<DMatch> good;
    unordered_map<int, int> m12;
    for (size_t i = 0; i < matches12.size(); ++i) {
        if (matches12[i].empty()) continue;
        int j = matches12[i][0].trainIdx;
        m12[(int)i] = j;
    }
    for (size_t j = 0; j < matches21.size(); ++j) {
        if (matches21[j].empty()) continue;
        int i = matches21[j][0].trainIdx;
        auto it = m12.find(i);
        if (it != m12.end() && it->second == (int)j) {
            DMatch dm;
            dm.queryIdx = i;
            dm.trainIdx = (int)j;
            dm.distance = matches12[i][0].distance;
            good.push_back(dm);
        }
    }
    return good;
}

int main() {
    // === 1.1 Charger l'image de référence ===
    string ref_path = "path.png";
    string video_path = "path.avi";

    Mat ref = imread(ref_path);
    if (ref.empty()) {
        cerr << "Erreur : impossible de charger l'image de référence : " << ref_path << endl;
        return -1;
    }

    Ptr<AKAZE> akaze = AKAZE::create();
    vector<KeyPoint> kp_ref;
    Mat desc_ref;
    akaze->detectAndCompute(ref, noArray(), kp_ref, desc_ref);
    cout << "Points AKAZE détectés dans l'image de référence : " << kp_ref.size() << endl;

    // === 1.2 Ouvrir la vidéo ===
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Erreur : impossible d'ouvrir la vidéo : " << video_path << endl;
        return -1;
    }

    int total_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);
    g_total_frames = max(1, total_frames);
    int fps = max(1, (int)round(cap.get(CAP_PROP_FPS)));
    cout << "Vidéo : " << total_frames << " frames | " << fps << " FPS" << endl;

    // === Préparer l’affichage ===
    const string win = "AR Tracker (AKAZE)";
    namedWindow(win, WINDOW_NORMAL);
    createTrackbar("Progression", win, &g_trackbar_pos, max(1, g_total_frames - 1), on_trackbar);

    BFMatcher matcher(NORM_HAMMING);
    bool paused = false;

    Mat frame;
    int frame_idx = 0;
    g_trackbar_pos = 0;
    cap.set(CAP_PROP_POS_FRAMES, 0);

    cout << "Contrôles : [Espace]=Pause/Play | [Échap]=Quitter | Glisser la barre pour naviguer\n";

    while (true) {
        if (!paused) {
            if (g_user_is_setting_trackbar) {
                int target = g_trackbar_pos;
                cap.set(CAP_PROP_POS_FRAMES, target);
                frame_idx = target;
                g_user_is_setting_trackbar = false;
            }

            bool ok = cap.read(frame);
            if (!ok || frame.empty()) {
                cap.set(CAP_PROP_POS_FRAMES, 0);
                frame_idx = 0;
                g_trackbar_pos = 0;
                setTrackbarPos("Progression", win, g_trackbar_pos);
                paused = true;

                Mat endImg = Mat::zeros(200, 600, CV_8UC3);
                putText(endImg, "Fin de la vidéo. Appuyez sur une touche pour redémarrer.",
                        Point(15, 100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2);
                imshow(win, endImg);
                int k = waitKey(0);
                if (k == 27) break;
                cap.set(CAP_PROP_POS_FRAMES, 0);
                frame_idx = 0;
                g_trackbar_pos = 0;
                setTrackbarPos("Progression", win, g_trackbar_pos);
                paused = false;
                continue;
            }
        } else {
            if (g_user_is_setting_trackbar) {
                int target = g_trackbar_pos;
                cap.set(CAP_PROP_POS_FRAMES, target);
                frame_idx = target;
                cap.read(frame);
                g_user_is_setting_trackbar = false;
            } else if (frame.empty()) {
                cap.read(frame);
            }
        }

        if (frame.empty()) continue;

        g_trackbar_pos = frame_idx;
        setTrackbarPos("Progression", win, g_trackbar_pos);

        auto t0 = chrono::high_resolution_clock::now();

        // === Détection AKAZE sur la frame ===
        vector<KeyPoint> kp_frame;
        Mat desc_frame;
        akaze->detectAndCompute(frame, noArray(), kp_frame, desc_frame);

        // === Appariement ===
        vector<DMatch> final_matches;
        bool homography_found = false;
        int good_matches_count = 0;
        Mat H;

        if (!desc_ref.empty() && !desc_frame.empty()) {
            vector<vector<DMatch>> knn12, knn21;
            matcher.knnMatch(desc_frame, desc_ref, knn12, 2);
            matcher.knnMatch(desc_ref, desc_frame, knn21, 1);

            const float ratio_thresh = 0.75f;
            vector<DMatch> good12;
            for (auto &k : knn12) {
                if (k.size() >= 2 && k[0].distance < ratio_thresh * k[1].distance)
                    good12.push_back(k[0]);
            }

            vector<vector<DMatch>> matches12_k1(desc_frame.rows);
            for (const DMatch &m : good12)
                matches12_k1[m.queryIdx] = vector<DMatch>(1, m);

            final_matches = symmetricMatches(matches12_k1, knn21);
            good_matches_count = (int)final_matches.size();

            if (good_matches_count >= 4) {
                vector<Point2f> pts_ref, pts_frame;
                for (auto &m : final_matches) {
                    pts_ref.push_back(kp_ref[m.trainIdx].pt);
                    pts_frame.push_back(kp_frame[m.queryIdx].pt);
                }

                vector<unsigned char> inliers;
                H = findHomography(pts_ref, pts_frame, RANSAC, 3.0, inliers);
                if (!H.empty() && countNonZero(inliers) >= 4) {
                    homography_found = true;
                }
            }
        }

        // === Affichage ===
        Mat display = frame.clone();
        if (homography_found) {
            vector<Point2f> ref_corners = {
                Point2f(0, 0),
                Point2f((float)ref.cols, 0),
                Point2f((float)ref.cols, (float)ref.rows),
                Point2f(0, (float)ref.rows)
            };
            vector<Point2f> scene_corners;
            perspectiveTransform(ref_corners, scene_corners, H);

            for (int i = 0; i < 4; i++) {
                line(display, scene_corners[i], scene_corners[(i + 1) % 4], Scalar(0, 255, 0), 3);
                circle(display, scene_corners[i], 4, Scalar(0, 0, 255), -1);
            }
        }

        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        string mode = homography_found ? "Mode: Tracking" : "Mode: Detection - " + to_string(good_matches_count) + " matches";
        putText(display, mode, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
        putText(display, "Temps: " + to_string((int)ms) + " ms", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);

        if (paused)
            putText(display, "PAUSE - Appuyez sur ESPACE pour reprendre", Point(10, 90),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);

        imshow(win, display);

        int key = waitKey(1);
        if (key == 27) break;         // Échap
        else if (key == 32) paused = !paused;  // Espace = Pause/Play

        if (!paused) ++frame_idx;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
