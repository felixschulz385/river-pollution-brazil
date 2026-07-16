#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

// Define a structure to represent a point in 2D space.
struct Point {
    double x, y; // Coordinates of the point
    
    // Constructor to initialize a Point. Default values are 0.
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    // Method to return the point's coordinates. (Note: This method currently has a non-functional return type and should return a pair or struct instead)
    double get() const {
        return(x, y);
    }
    
    // Method to return the point's coordinates as a string.
    std::string getString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }
};

// Function to calculate orthogonal points from a line segment defined by points A and B, at a specified distance.
std::pair<Point, Point> calculateOrthogonalPoints(const Point& A, const Point& B, double distance) {
    double dx = B.x - A.x; // Change in x
    double dy = B.y - A.y; // Change in y
    double magnitude = std::sqrt(dx*dx + dy*dy); // Magnitude of vector AB
    double dx_normalized = dx / magnitude; // Normalized change in x
    double dy_normalized = dy / magnitude; // Normalized change in y
    // Calculating the points above and below the line segment AB at the given distance
    Point orthogonalVectorAbove(dy_normalized, -dx_normalized);
    Point orthogonalVectorBelow(-dy_normalized, dx_normalized);
    Point pointAbove(B.x + distance * orthogonalVectorAbove.x, B.y + distance * orthogonalVectorAbove.y);
    Point pointBelow(B.x + distance * orthogonalVectorBelow.x, B.y + distance * orthogonalVectorBelow.y);
    
    return {pointAbove, pointBelow}; // Return the points above and below
};

// Function to calculate the midway point between two points A and B, adjusted away from an anchor point by a given distance.
Point calculateMidwayPoint(const Point& A, const Point& B, const Point& Anchor, double distance) {
    double xh = (A.x + B.x) / 2; // Midpoint x-coordinate
    double yh = (A.y + B.y) / 2; // Midpoint y-coordinate
    // Calculate the vector from the anchor to the midpoint
    double xd = (xh - Anchor.x);
    double yd = (yh - Anchor.y);
    double magnitude = std::sqrt(xd*xd + yd*yd); // Magnitude of this vector
    double xd_normalized = xd / magnitude; // Normalized change in x
    double yd_normalized = yd / magnitude; // Normalized change in y
    // Return the adjusted midway point
    return Point(xh + distance * xd_normalized, yh + distance * yd_normalized);
};

// Function to calculate the angle between two vectors in radians
double angleBetweenVectors(const Point& vectorA, const Point& vectorB) {
    double dotProduct = vectorA.x * vectorB.x + vectorA.y * vectorB.y;
    double magnitudeA = sqrt(vectorA.x * vectorA.x + vectorA.y * vectorA.y);
    double magnitudeB = sqrt(vectorB.x * vectorB.x + vectorB.y * vectorB.y);
    double angle = acos(dotProduct / (magnitudeA * magnitudeB));
    // Print angle for debugging
    std::cout << "Angle: " << angle << std::endl;
    return angle; // Angle in radians
}

double adjustDistanceBasedOnAngle(const Point& A, const Point& B, const Point& C, double baseDistance) {
    Point vectorAB = {B.x - A.x, B.y - A.y};
    Point vectorBC = {C.x - B.x, C.y - B.y};
    double angle = angleBetweenVectors(vectorAB, vectorBC);

    // Angle threshold in radians, for example 45 degrees = PI/4
    const double threshold = M_PI / 4;
    // Scale factor to reduce distance for sharp angles
    const double scaleFactor = 0.5;

    // Adjust the distance based on the angle between the vectors AB and BC
    if (angle > threshold) {
        std::cout << "Adjusting distance" << std::endl;
        return std::max(baseDistance * scaleFactor, baseDistance * (threshold / angle));
    }
    return baseDistance;
}

extern "C" {

    void calculatePolygons(const double* X, const double* Y, int N, double distance, double** polygonsX, double** polygonsY) {
        std::vector<std::pair<Point, Point>> midwayPointsList(N-2);

        // Redirect stdout to a file
        freopen("output.txt", "a", stdout);

        for(int i = 1; i < N - 1; ++i) {
            // Calculate orthogonal points for the line segment between (X[i-1], Y[i-1]) and (X[i], Y[i])
            Point A(X[i-1], Y[i-1]), B(X[i], Y[i]), C(X[i+1], Y[i+1]);

            // Adjust the distance based on the angle between the vectors AB and BC
            double cdistance = adjustDistanceBasedOnAngle(A, B, C, distance);

            // Calculate the orthogonal points
            std::pair<Point, Point> orthogonalPointsAB = calculateOrthogonalPoints(A, B, cdistance);
            std::pair<Point, Point> orthogonalPointsCB = calculateOrthogonalPoints(C, B, cdistance);

            // Calculate the midway point between the orthogonal points
            Point midwayPointAB = calculateMidwayPoint(orthogonalPointsAB.first, orthogonalPointsCB.second, B, cdistance);
            Point midwayPointCB = calculateMidwayPoint(orthogonalPointsCB.first, orthogonalPointsAB.second, B, cdistance);

            // Store the points
            midwayPointsList[i-1] = std::make_pair(midwayPointAB, midwayPointCB);
        }

        // Allocate memory for the output arrays
        *polygonsX = (double*)malloc((N - 3) * sizeof(double) * 6);
        *polygonsY = (double*)malloc((N - 3) * sizeof(double) * 6);

        for(int i = 0; i < N - 3; ++i) {
            (*polygonsX)[i * 6 + 0] = midwayPointsList[i].first.x;
            (*polygonsY)[i * 6 + 0] = midwayPointsList[i].first.y;
            (*polygonsX)[i * 6 + 1] = X[i+1];
            (*polygonsY)[i * 6 + 1] = Y[i+1];
            (*polygonsX)[i * 6 + 2] = midwayPointsList[i].second.x;
            (*polygonsY)[i * 6 + 2] = midwayPointsList[i].second.y;
            (*polygonsX)[i * 6 + 3] = midwayPointsList[i+1].second.x;
            (*polygonsY)[i * 6 + 3] = midwayPointsList[i+1].second.y;
            (*polygonsX)[i * 6 + 4] = X[i+2];
            (*polygonsY)[i * 6 + 4] = Y[i+2];
            (*polygonsX)[i * 6 + 5] = midwayPointsList[i+1].first.x;
            (*polygonsY)[i * 6 + 5] = midwayPointsList[i+1].first.y;
        }
    }
}
