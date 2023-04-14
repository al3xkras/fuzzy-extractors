### Fuzzy Extractors Course project

1. Project description
   - Purpose

      The purpose of the project is to create a Fuzzy extractor based on a pre-built face recognition model.
      As the pre-built model, @ageitgey's face_recognition model is used: https://github.com/ageitgey/face_recognition
   
      The FuzzyExtractor class requirements:
      - Should receive a set of images (e.g. video or a set of photos) as the input data
      - Should preprocess images: crop each face rectangle, remove duplicate images etc
      - Should calculate the p-value of the test: whether each image of the given set includes the same person (and does not include >1 people)
      - Should create a primary hash value, based on the p-value, and cropped images
         - The hash function should be collision resistant 
          (hash value of similar within the confidence level faces should be very similar, but not equal)
         - Given a hash value, it should not be possible to retrieve the actual face landmarks
           (first image resistance)
         - The hash is not required to be 2-nd image resistant: 
          (given face image, it is always possible 
          to find a different face image with equal or similar hash)
      - create a secondary hash value, based on the primary hash value: (possible algorithms: SHA256)
         - map similar within a confidence interval primary hash codes to equal secondary codes
         - collision resistance for hash codes, that are not similar within the confidence level
         - first image resistance
         - second image resistance
      - expand the hash value to the required key length
      - 
   - Implemented use cases:
     1. User wants to encode some file using the AES algorithm.  
      A key for the algorithm is generated using the 
      FuzzyExtractor class. The user does not know the 
      password, but to decode the file, it is only required 
      to complete a face recognition test
     
     2. User wants to generate a key pair for the ECDSA algorithm to sign a document.
      Using the FuzzyExtractor class, it is possible to generate a key pair based on the user's face landmarks (+salt).
      If user would like then to sign another document, (without having the actual private and public key),
      it will be much easier, as user will only have to complete a face recognition test.
   
     3. User wants to access a certain web-resource. In order to authenticate the user, 
      the service asks them to submit their private key, which is generated using biometric data.
   

2. Project structure
   - a
   - b
   - c


3. Use cases

4Refs

    (0) https://en.m.wikipedia.org/wiki/Fuzzy_extractor

    (1) https://en.m.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction
 
    (2) https://arxiv.org/abs/cs/0602007
    
    (3) https://digital.csic.es/bitstream/10261/15966/1/SAM3262.pdf
    
    (4) https://www.arijuels.com/wp-content/uploads/2013/09/JS02.pdf
    
    (5) https://ro.uow.edu.au/cgi/viewcontent.cgi?article=1698&context=eispapers1
    
    (6) https://github.com/ageitgey/face_recognition

    (7) https://stackoverflow.com/questions/70492290/how-to-make-a-face-recognitionidentification-and-verification-program
