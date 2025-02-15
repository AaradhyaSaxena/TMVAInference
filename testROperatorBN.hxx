#ifndef TEST_OPERATOR_BN
#define TEST_OPERATOR_BN

#include "TMVA/RTensor.hxx"
#include "testROperator.hxx"
#include "ROperatorBN.hxx"
#include <sstream>

template<typename T>
bool testROperatorBNex1(double tol);

template<typename T>
bool testROperatorBN(double tol) {
   bool failed = false;

   failed |= testROperatorBNex1<T>(tol);
   
   return failed;
}


template<typename T>
bool testROperatorBNex1(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorBN;

   // X
   T inputX[120] = {-0.7977636 ,  1.1958243 , -1.0229037 , -0.51732904,  0.12441001,
        1.9109865 , -0.38977718,  0.94658875, -1.2673457 , -0.07290398,
       -0.6696631 , -0.7541014 , -1.8153696 , -0.377495  ,  0.14793058,
        0.6856624 , -0.12023604, -2.6498742 , -0.08928333,  0.49031177,
       -0.47636536, -0.73035645, -0.6218881 , -0.47484934,  0.39107057,
        0.13585244, -1.7441463 , -0.7515306 ,  0.3404593 , -0.13743791,
        0.96228236, -1.8259237 ,  1.2595884 , -0.10568807,  0.20880911,
        0.08542808,  0.3833048 ,  0.16968544,  2.6749225 ,  0.39770764,
        1.1793861 , -0.28413948, -1.6303368 , -0.3517759 , -0.9084473 ,
        2.3107078 , -0.05392435, -0.09970386, -1.4834313 ,  0.81488115,
        0.94946486,  0.5208065 ,  0.6334754 ,  0.27497092,  0.8291302 ,
        1.048831  ,  0.36349958,  0.6671382 ,  0.47251478, -1.121705  ,
        0.26652038, -2.776358  ,  1.4315637 , -1.1211625 , -1.6951979 ,
       -1.4067265 ,  0.87892413, -0.15322469, -2.023544  , -0.35543326,
        2.5225415 ,  0.540655  ,  0.01237731, -0.6971843 ,  1.5347829 ,
        1.0074348 ,  1.5320773 ,  0.05634216, -1.1590399 ,  0.64761645,
       -0.2426215 ,  1.8763269 , -0.20529938, -1.4458609 ,  1.4042926 ,
       -0.19217268, -0.32174838,  0.3894971 , -0.9743113 , -0.09593748,
       -0.10383652,  0.636563  ,  0.55245966,  1.0055209 ,  0.85369015,
        0.4306947 ,  0.1850241 ,  1.4003046 ,  1.5885189 , -0.4484351 ,
       -0.36964896, -1.8713006 , -0.31819174,  0.38620153, -0.17096515,
        1.5749574 ,  0.36125645, -1.2688363 ,  1.2529895 , -1.8408806 ,
       -0.33599356, -1.1130788 , -1.4131204 ,  0.29703423,  0.5716227 ,
       -1.5414467 , -0.51876396,  1.4948657 ,  0.26339644,  0.01545961};
	RTensor<T> X(inputX, {2, 3, 4, 5});	

   // s
   T inputS[3] = {0.46402985,  0.9471513 , -0.69440895};
   RTensor<T> s(inputS, {3});

   // bias
   T inputBias[3] = { -0.7263242,  0.3700269, -1.2367871};
   RTensor<T> bias(inputBias, {3});

   // mean
   T inputMean[3] = { 0.31705657,  0.15278828, -0.7104717};
   RTensor<T> mean(inputMean, {3});

   // var
   T inputVar[3] = { 0.21115398, 0.24694446, 0.219496};
   RTensor<T> var(inputVar, {3});
   
   // Y
   T inputY[120] = {-1.85207117e+00,  1.61056697e-01, -2.07941794e+00, -1.56888807e+00,
       -9.20858979e-01,  8.83228660e-01, -1.44008589e+00, -9.06216502e-02,
       -2.32625580e+00, -1.12010694e+00, -1.72271514e+00, -1.80798101e+00,
       -2.87965155e+00, -1.42768335e+00, -8.97107840e-01, -3.54105473e-01,
       -1.16790295e+00, -3.72233534e+00, -1.13664675e+00, -5.51370859e-01,
       -8.29106867e-01, -1.31320071e+00, -1.10646558e+00, -8.26217592e-01,
        8.24180424e-01,  3.37748051e-01, -3.24543118e+00, -1.35355735e+00,
        7.27717996e-01, -1.83129013e-01,  1.91288018e+00, -3.40129447e+00,
        2.47952986e+00, -1.22615486e-01,  4.76799637e-01,  2.41641894e-01,
        8.09379280e-01,  4.02231991e-01,  5.17708254e+00,  8.36830318e-01,
       -4.03783941e+00, -1.86867523e+00,  1.26590729e-01, -1.76842797e+00,
       -9.43357527e-01, -5.71462727e+00, -2.20988870e+00, -2.14203644e+00,
       -9.11451578e-02, -3.49758840e+00, -3.69706178e+00, -3.06172562e+00,
       -3.22871828e+00, -2.69736075e+00, -3.51870751e+00, -3.84433722e+00,
       -2.82857347e+00, -3.27861142e+00, -2.99015045e+00, -6.27277613e-01,
       -7.77355731e-01, -3.85005903e+00,  3.99106741e-01, -2.17863989e+00,
       -2.75830173e+00, -2.46700287e+00, -1.58949494e-01, -1.20121491e+00,
       -3.08986616e+00, -1.40540540e+00,  1.50077760e+00, -5.00534177e-01,
       -1.03398979e+00, -1.75050616e+00,  5.03337741e-01, -2.91792154e-02,
        5.00605464e-01, -9.89593983e-01, -2.21688843e+00, -3.92524362e-01,
       -3.83603394e-01,  3.65500116e+00, -3.12469363e-01, -2.67691469e+00,
        2.75532818e+00, -2.87450612e-01, -5.34415066e-01,  8.21181476e-01,
       -1.77816558e+00, -1.04031384e-01, -1.19086534e-01,  1.29207611e+00,
        1.13177955e+00,  1.99529052e+00,  1.70590901e+00,  8.99701893e-01,
        4.31466669e-01,  2.74772739e+00,  3.10645390e+00, -7.75873244e-01,
       -1.74193740e+00,  4.83735204e-01, -1.81820464e+00, -2.86222124e+00,
       -2.03641653e+00, -4.62413502e+00, -2.82524872e+00, -4.09207106e-01,
       -4.14693069e+00,  4.38648343e-01, -1.79181981e+00, -6.40062988e-01,
       -1.95356369e-01, -2.73006177e+00, -3.13704324e+00, -5.15758991e-03,
       -1.52092671e+00, -4.50542736e+00, -2.68020558e+00, -2.31272602e+00}; 
   
   RTensor<T> TrueY(inputY, {2, 3, 4, 5});
   RTensor<T> Y({2, 3, 4, 5});

   ROperatorBN<T> bn(  1e-5,      /* epsilon */
                       0.9,       /* momentum */
                       0);        /* training_mode */

   bn.Forward_blas(X, s, bias, mean, var, Y);
   // std::cout<<Y<<std::endl;

   bool failed = !IsApprox(Y, TrueY, tol);

   std::stringstream ss;
   ss << "   ";
   ss << "Batch Normalization ex1 : Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
}

#endif
