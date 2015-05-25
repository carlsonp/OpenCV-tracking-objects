CC = gcc


EXEC = task_two

FLAGS = -O0 -g3 -Wno-write-strings -Wall
FILES = task_two.cpp
OBJS = task_two.o

LIBS = -lcxcore -lcv -lhighgui -lcvaux -lml

all:
	$(CC) $(FLAGS) -c $(FILES) -I /usr/include/opencv -L /usr/lib
	$(CC) $(FLAGS) -o $(EXEC) $(OBJS) $(LIBS) -I /usr/include/opencv -L /usr/lib
	rm *.o
	@echo "Done!"
