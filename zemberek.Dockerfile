FROM eclipse-temurin:8

WORKDIR /app

RUN mkdir zemberek_data 
COPY ./zemberek_data ./zemberek_data

COPY zemberek-full.jar .



CMD ["java", "-jar", "zemberek-full.jar", "StartGrpcServer", "--dataRoot", "./zemberek_data/"]