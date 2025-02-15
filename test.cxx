#include <iostream>

#include <TMVA/RTensor.hxx>

#include "testROperatorConv.hxx"
#include "testROperatorBN.hxx"
#include "testROperatorIN.hxx"

int main(int argc, char *argv[]) {
   if (argc != 2) {
      std::cout << "+---------------------------------------------+" << std::endl;
      std::cout << "| ./test operator_name                        |" << std::endl;
      std::cout << "+---------------------------------------------+" << std::endl;
      return 0;
   }

   std::string operator_name(argv[1]);
   std::cout << "Testing operator " << operator_name << std::endl;
   bool failed = false;

   if (operator_name == "BN") {
      std::cout << "-----------------------------------------------------------------" << std::endl;
      failed |= testROperatorBN<float>(1.E-3);
      std::cout << (failed? "Test Failed" : "Test Passed" ) << std::endl;

      return failed;
   }
   else if (operator_name == "IN") {
      std::cout << "-----------------------------------------------------------------" << std::endl;
      failed |= testROperatorIN<float>(1.E-2);
      std::cout << (failed? "Test Failed" : "Test Passed" ) << std::endl;

      return failed;
   }

   else if (operator_name == "Conv") {
      std::cout << "-----------------------------------------------------------------" << std::endl;
      failed |= testROperatorConv<float>(1.E-3);
      std::cout << (failed? "Test Failed" : "Test Passed" ) << std::endl;

      return failed;
   }
}
