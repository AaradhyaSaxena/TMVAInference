#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_IN
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_IN

#include <TMVA/RTensor.hxx>
#include <iostream>

//#include "Blas.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// Operator IN
template<typename T> class ROperatorIN {
private:
   /* Attributes */
   T fepsilon = 1e-05;

public:
   /* Constructor */
   ROperatorIN( T epsilon):
      fepsilon(epsilon){}

   /* Forward pass using Blas */
   void Forward_blas(const RTensor<T> &X,
                     const RTensor<T> &scale,
                     const RTensor<T> &B,
                           RTensor<T> &Y);

};

template<typename T>
void ROperatorIN<T>::Forward_blas(	const RTensor<T> &X,
									const RTensor<T> &scale,
									const RTensor<T> &B,
										RTensor<T> &Y) {
   
   // inference mode
   std::size_t inputSize = X.GetShape().size();
	if (inputSize == 4) { 
		// Input has size batchSize x channels x height x width
		std::size_t batchSize = X.GetShape()[0];
		std::size_t channels  = X.GetShape()[1];
		std::size_t height    = X.GetShape()[2];
		std::size_t width     = X.GetShape()[3];
		static const int n = batchSize * channels * height * width;

		// IN op
		for (std::size_t b = 0; b < batchSize; b++) {
			for (std::size_t c = 0; c < channels; c++) {
				double mean = 0, var = 0;
				for (std::size_t h = 0; h < height; h++) {
					for (std::size_t w = 0; w < width; w++) {
						mean +=  X(b, c, h, w);
					}
				}
				mean = mean/(height*width);
				for (std::size_t h = 0; h < height; h++) {
					for (std::size_t w = 0; w < width; w++) {
						var +=  (X(b, c, h, w)- mean)*(X(b, c, h, w)- mean);
					}
				}
				var = var/(height*width);
				for (std::size_t h = 0; h < height; h++) {
					for (std::size_t w = 0; w < width; w++) {
						Y(b, c, h, w) = (((X(b, c, h, w) - mean)/ sqrt(var + fepsilon)) * scale(c)) + B(c);
					}
				}
			}
		}
	}	   
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
