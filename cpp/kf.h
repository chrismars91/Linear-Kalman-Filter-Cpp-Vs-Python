#pragma once

#include <Eigen/Dense>
// #include <iostream>

class KF
{
public:

    using M6 = Eigen::Matrix<double, 6, 6>;
    using M61 = Eigen::Matrix<double, 6, 1>;
    using M36 = Eigen::Matrix<double, 3, 6>;
    using M3 = Eigen::Matrix<double, 3, 3>;
    using M31 = Eigen::Matrix<double, 3, 1>;
    using M63 = Eigen::Matrix<double, 6, 3>;

    M61 x;
    M6 P;
    M6 Q;
    M6 F;
    M36 H;
    M3 R;
    M31 z;
    M63 K;
    M31 y;
    M3 S;
    M3 SI;

    M6 _I;
    M63 PHT;
    M6 I_KH;

    KF(double sensor_noise_std, double processes_model_std)
    {

    /*
    :param sensor_noise_std: what is the expected error in your sensor? this value is easy, how much can
           you trust your sensor.
    :param processes_model_std: what is expected error in your model? This is a linear model, look at the F matrix.
           If the object has any acceleration, the model will need some error to describe it. If the model had
           constant velocity than this KF would be perfect and all your error would be in the sensor or
           sensor_noise_var. Finding the best value can be a trial and error processes.
    */

        P << 
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, .10, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, .10, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, .10;

        H << 
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0;

        x.setZero();
        Q.setIdentity();
        F.setIdentity();
        R.setIdentity();
        K.setZero();
        y.setZero();
        S.setZero();
        SI.setZero();

        R *= (sensor_noise_std*sensor_noise_std);
        Q *= (processes_model_std*processes_model_std);

        _I.setIdentity();
        PHT.setZero();
        I_KH.setZero();

    }

    void predict(double dt)
    {
        F(0, 3) = dt;
        F(1, 4) = dt;
        F(2, 5) = dt;
        x = F*x;
        P = ((F*P)*F.transpose()) + Q;
    }

    void update(M31 z)
    {
        y = z - (H*x);
        PHT = P * H.transpose();
        S = (H*PHT) + R;
        S.inverse();
        SI = S.inverse();
        K = PHT*SI;
        x = x+(K*y);
        I_KH = _I - (K*H);
        P = ((I_KH*P)*I_KH.transpose()) + ((K*R)*K.transpose());
    }

    M61 get_prediction(double dt)
    {
        F(0, 3) = dt;
        F(1, 4) = dt;
        F(2, 5) = dt;
        return F * x;
    }

    M63 get_K() const
    {
        return K;
    }


    M6 get_P() const
    {
        return P;
    }


    Eigen::MatrixXd create_matrix(int rows, int cols) {
        return Eigen::MatrixXd::Random(rows, cols);
    }

private:

};