#include <iostream>
#include <vector>
#include <cmath>

struct Point {
    double x, y;
    
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    double get() const {
        return(x, y);
    }
    std::string getString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }
};

std::pair<Point, Point> calculateOrthogonalPoints(const Point& A, const Point& B, double distance) {
    double dx = B.x - A.x;
    double dy = B.y - A.y;
    double magnitude = std::sqrt(dx*dx + dy*dy);
    double dx_normalized = dx / magnitude;
    double dy_normalized = dy / magnitude;
    Point orthogonalVectorAbove(dy_normalized, -dx_normalized);
    Point orthogonalVectorBelow(-dy_normalized, dx_normalized);
    Point pointAbove(B.x + distance * orthogonalVectorAbove.x, B.y + distance * orthogonalVectorAbove.y);
    Point pointBelow(B.x + distance * orthogonalVectorBelow.x, B.y + distance * orthogonalVectorBelow.y);
    
    return {pointAbove, pointBelow};
}

Point calculateMidwayPoint(const Point& A, const Point& B) {
    return Point((A.x + B.x) / 2, (A.y + B.y) / 2);
}

int main() {
    // Example arrays X and Y
    std::vector<double> X = {1, 4, 6, 8};
    std::vector<double> Y = {2, 6, 7, 9};
    double distance = 5;
    int N = X.size(); // Assuming X and Y are of the same size

    std::vector<std::pair<Point, Point>> midwayPointsList(N-2);

    for(int i = 1; i < N - 1; ++i) {
        // Calculate orthogonal points for the line segment between (X[i-1], Y[i-1]) and (X[i], Y[i])
        Point A(X[i-1], Y[i-1]), B(X[i], Y[i]), C(X[i+1], Y[i+1]);
        std::pair<Point, Point> orthogonalPointsAB = calculateOrthogonalPoints(A, B, distance);
        std::pair<Point, Point> orthogonalPointsCB = calculateOrthogonalPoints(C, B, distance);

        // Calculate the midway point between the orthogonal points
        Point midwayPointAB = calculateMidwayPoint(orthogonalPointsAB.first, orthogonalPointsCB.second);
        Point midwayPointCB = calculateMidwayPoint(orthogonalPointsCB.first, orthogonalPointsAB.second);

        // Store the points
        midwayPointsList[i-1] = std::make_pair(midwayPointAB, midwayPointCB);
        /*
        std::cout << "OP: " << i << std::endl;
        std::cout << orthogonalPointsAB.first.getString() << ", " << orthogonalPointsAB.second.getString() << std::endl;
        std::cout << orthogonalPointsCB.first.getString() << ", " << orthogonalPointsCB.second.getString() << std::endl;
        std::cout << midwayPointAB.getString() << ", " << midwayPointCB.getString() << std::endl;
        */
    }

    std::vector<std::vector<double>> polygonsX(N - 3, std::vector<double>(6));
    std::vector<std::vector<double>> polygonsY(N - 3, std::vector<double>(6));
    
    for(int i = 0; i < N - 3; ++i) {
        polygonsX[i][0] = midwayPointsList[i].first.x;
        polygonsY[i][0] = midwayPointsList[i].first.y;
        polygonsX[i][1] = X[i+1];
        polygonsY[i][1] = Y[i+1];
        polygonsX[i][2] = midwayPointsList[i].second.x;
        polygonsY[i][2] = midwayPointsList[i].second.y;
        polygonsX[i][3] = midwayPointsList[i+1].second.x;
        polygonsY[i][3] = midwayPointsList[i+1].second.y;
        polygonsX[i][4] = X[i+2];
        polygonsY[i][4] = Y[i+2];
        polygonsX[i][5] = midwayPointsList[i+1].first.x;
        polygonsY[i][5] = midwayPointsList[i+1].first.y;
        
        /*
        std::cout << "Polygon: " << i << std::endl;
        for(int j = 0; j < 6; ++j) {
            std::cout << "(" << polygonsX[i][j] << ", " << polygonsY[i][j] << ")" << std::endl;
        }
        */
    }
    return 0;
}
