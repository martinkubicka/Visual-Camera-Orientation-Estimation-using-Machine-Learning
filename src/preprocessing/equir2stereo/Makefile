OUTPUT_DIR = out

all:
	@g++ src/main.cpp -o src/main

run:
	@if [ ! -d $(OUTPUT_DIR) ]; then mkdir $(OUTPUT_DIR); fi
	@./src/main assets/front.png assets/right.png assets/back.png assets/left.png assets/top.png assets/down.png

.PHONY: clean
clean:
	@rm src/main
