import argparse
import bilateral_detector
import symmetry_drawer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect symmetry in images')
    parser.add_argument("--save_feature_points", type=str)
    parser.add_argument("--save_hexbin", type=str)
    parser.add_argument("--save_matchpoints", type=str)
    parser.add_argument("-o", "--out", type=str, required=True)
    parser.add_argument("-s", "--source", type=str, required=True)
    args = parser.parse_args()
    bilateral_detector = bilateral_detector.BilateralDetecotor()
    symmetry_drawer = symmetry_drawer.SymmetryDrawer()
    bilateral_detector.find(args.source)
    if args.save_feature_points is not None:
        symmetry_drawer.draw_keypoints(args.save_feature_points, bilateral_detector)
    if args.save_hexbin is not None:
        symmetry_drawer.draw_hexbin(args.save_hexbin, bilateral_detector)
    if args.save_matchpoints is not None:
        symmetry_drawer.draw_matchpoints(args.save_matchpoints, bilateral_detector)
    symmetry_drawer.draw_symmetry(args.out, bilateral_detector)