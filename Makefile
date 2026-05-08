# Compiler settings
CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LIBS = -lm

# Project files
TARGET = ascp
SRC = ascp.c
HEADERS = stb_image.h stb_image_write.h stb_image_resize2.h

# Build rules
all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

# Utility to clean build artifacts
clean:
	rm -f $(TARGET)

# Run a test (assuming you have a test image named 'input.jpg')
test: all
	./$(TARGET) input.png output_enhanced.png
	
.PHONY: all clean test