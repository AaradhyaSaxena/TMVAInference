#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_BN
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_BN

#include <TMVA/RTensor.hxx>
#include <iostream>

//#include "Blas.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// Operator BN
template<typename T> class ROperatorBN {
private:
   /* Attributes */
   T fepsilon = 1e-05;
   T fmomentum = 0.9;
   std::size_t ftraining_mode = 0;

public:
   /* Constructor */
   ROperatorBN( T epsilon, T momentum, std::size_t training_mode):
      fepsilon(epsilon), fmomentum(momentum), ftraining_mode(training_mode){}

   /* Forward pass using Blas */
   void Forward_blas(const RTensor<T> &X,
                     const RTensor<T> &scale,
                     const RTensor<T> &B,
                           RTensor<T> &input_mean,
                           RTensor<T> &input_var,
                           RTensor<T> &Y);

};

template<typename T>
void ROperatorBN<T>::Forward_blas(	const RTensor<T> &X,
									const RTensor<T> &scale,
									const RTensor<T> &B,
										RTensor<T> &input_mean,
										RTensor<T> &input_var,
										RTensor<T> &Y) {
   
   // inference mode
   std::size_t inputSize = X.GetShape().size();
   if (ftraining_mode == 0) {
	   if (inputSize == 4) { 
		   // Input has size batchSize x channels x height x width
		   std::size_t batchSize = X.GetShape()[0];
		   std::size_t channels  = X.GetShape()[1];
		   std::size_t height    = X.GetShape()[2];
		   std::size_t width     = X.GetShape()[3];

		   // Initialize output vector
		   // RTensor<T> Y({batchSize, channels, height, width});

		   // Batch normalization operation
		   std::size_t normalized_x = 0;
		   for (std::size_t b = 0; b < batchSize; b++) {
				for (std::size_t c = 0; c < channels; c++) {
					for (std::size_t h = 0; h < height; h++) {
						for (std::size_t w = 0; w < width; w++) {
							normalized_x = (X(b, c, h, w) - input_mean(c))/ sqrt(input_var(c) + fepsilon);
							Y(b, c, h, w) = (normalized_x * scale(c)) + B(c);
						}
					}
				}
			}
	   }	   
   }
   else{
	   std::stringstream ss;
	   ss << "TMVA::SOFIE - training_mode not supported";
	   ss << inputSize;
	   throw std::runtime_error(ss.str());
   }
  
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
