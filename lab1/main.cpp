#include <iostream>
#include <chrono>

// xk+1 = xk + 3xk(1-xk) x0=0.1
// xk+1 = 4xk - 3xkxk

using namespace std;

int main() {
    int k = 70;

    cout << "float 1 : [0.1";
    float float1 = 0.1;
    for (int i = 1; i<=k; i++) {
        cout << ", ";
        float1 = float1 + 3 * float1 * (1 - float1);
        cout << float1;
    }
    cout << "]\n\n";

    cout << "float 2 : [0.1";
    float float2 = 0.1;
    for (int i = 1; i<=k; i++) {
        cout << ", ";
        float2 = float2 * 4 - 3 * float2 * float2;
        cout << float2;
    }
    cout << "]\n\n";

    cout << "double 1 : [0.1";
    double double1 = 0.1;
    for (int i = 1; i<=k; i++) {
        cout << ", ";
        double1 = double1 + 3 * double1 * (1 - double1);
        cout << double1;
    }
    cout << "]\n\n";

    cout << "double 2 : [0.1";
    double double2 = 0.1;
    for (int i = 1; i<=k; i++) {
        cout << ", ";
        double2 = double2 * 4 - 3 * double2 * double2;
        cout << double2;
    }
    cout << "]\n\n";

    cout << "long double 1 : [0.1";
    long double long_double1 = 0.1;
    for (int i = 1; i<=k; i++) {
        cout << ", ";
        long_double1 = long_double1 + 3 * long_double1 * (1 - long_double1);
        cout << long_double1;
    }
    cout << "]\n\n";

    cout << "long double 2 : [0.1";
    long double long_double2 = 0.1;
    for (int i = 1; i<=k; i++) {
        cout << ", ";
        long_double2 = long_double2 * 4 - 3 * long_double2 * long_double2;
        cout << long_double2;
    }
    cout << "]\n\n";




    cout << "\nTIME MEASURMENTS:\n ";

    auto start = chrono::steady_clock::now();
    auto end = chrono::steady_clock::now();

    cout << "float 1 : ";
    float1 = 0.1;
    start = chrono::steady_clock::now();
    for (int i = 1; i<=k; i++) {
        float1 = float1 + 3 * float1 * (1 - float1);
    }
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
         << " ns" << endl;

    cout << "float 2 : ";
    float2 = 0.1;
    start = chrono::steady_clock::now();
    for (int i = 1; i<=k; i++) {
        float2 = float2 * 4 - 3 * float2 * float2;
    }
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
         << " ns" << endl;

    cout << "double 1 : ";
    double1 = 0.1;
    start = chrono::steady_clock::now();
    for (int i = 1; i<=k; i++) {
        double1 = double1 + 3 * double1 * (1 - double1);
    }
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
         << " ns" << endl;


    cout << "double 2 : ";
    double2 = 0.1;
    start = chrono::steady_clock::now();
    for (int i = 1; i<=k; i++) {
        double2 = double2 * 4 - 3 * double2 * double2;
    }
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
         << " ns" << endl;


    cout << "long double 1 : ";
    long_double1 = 0.1;
    start = chrono::steady_clock::now();
    for (int i = 1; i<=k; i++) {
        long_double1 = long_double1 + 3 * long_double1 * (1 - long_double1);
    }
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
         << " ns" << endl;


    cout << "long double 2 : ";
    long_double2 = 0.1;
    start = chrono::steady_clock::now();
    for (int i = 1; i<=k; i++) {
        long_double2 = long_double2 * 4 - 3 * long_double2 * long_double2;
    }
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
         << " ns" << endl;


    printf("%ld", sizeof(long_double1));

}

