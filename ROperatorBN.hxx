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
		   static const int n = batchSize * channels * height * width;

			//// BN Blas implmentation			
			// Intialize A
			T* A = nullptr;
			A = new T[channels];
			for (std::size_t c = 0; c < channels; c++) {
				A[c] = scale(c)/ sqrt(input_var(c) + fepsilon);
			}

			// Broadcast A, bias and input_mean to shape_X
			T* Ba = nullptr;
			Ba = new T[n];
			T* Bmean = nullptr;
			Bmean = new T[n];
			T* Bbias = nullptr;
			Bbias = new T[n];

			size_t bs = 0, ch = 0, h = 0, w = 0;
			for(ch=0; ch<channels; ch++){
				for(h=0; h<height; h++){
					for(w=0; w<width; w++){
						Ba[bs*channels*height*width + ch*height*width + h*width + w] = A[ch];
						Bmean[bs*channels*height*width + ch*height*width + h*width + w] = input_mean(ch);
						Bbias[bs*channels*height*width + ch*height*width + h*width + w] = B(ch);
					}
				}
			}
			size_t Batchoffset = channels*height*width;
			for(bs = 1; bs<batchSize; bs++){
				std::copy(Ba, Ba+Batchoffset, Ba+(bs*Batchoffset));
				std::copy(Bmean, Bmean+Batchoffset, Bmean+(bs*Batchoffset));
				std::copy(Bbias, Bbias+Batchoffset, Bbias+(bs*Batchoffset));
			}
			
			// Initialize C with X
			T* C = nullptr;
			C = new T[n];
			std::copy(X.GetData(), (X.GetData()+n), C);

			/// blas saxpy (C = X - Bmean)
			int incx = 1;
			int incy = 1;
			float alpha = -1.;
			BLAS::saxpy_(&n, &alpha, Bmean, &incx, C, &incy);
			
			// blas smbv (Y = CxBa + Bbias)
			static const int k = 0; 
			static const double alpha2 = 1.0;
			static const int lda = 1;
			static const double beta = 1;
			incx = 1; incy = 1;

			//sbmv
			// BLAS::dsbmv_("L", &n, &k, &alpha2, C, &lda, Ba, &incx, &beta, Bbias, &incy);

			// sdot
			// T* temp = nullptr;
			// temp = new T[n];
			// temp = BLAS::sdot_(&n, C, &incx, Ba, &incy);

			// Y = CxBa + Bbias) and Y = Bbias;
			for(std::size_t i=0; i<n; i++){
				Y((i/(channels*height*width)%batchSize), (i/(height*width)%channels), (i/(width)%height), (i%width)) = C[i]*Ba[i] + Bbias[i];
				// std::cout<<Y((i/(channels*height*width)%batchSize), (i/(height*width)%channels), (i/(width)%height), (i%width))<<" ";
			}
			// std::cout<<std::endl;
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
