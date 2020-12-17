# TARGET
TARGET 				= 		darknet_plantform

#LIB
PKG					=		\

# ENV
SRC_DIR 			= 		./src

SOURCE 				= 		$(wildcard $(SRC_DIR)/*.cpp)

OBJ_DIR				= 		./obj

OBJECT 				= 		$(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(SOURCE)))

INCLUDE_DIR			=		./include

LIB_DIR				=		./lib

ALIB				=		$(LIB_DIR)/libdarknet.a
SLIB				=		$(LIB_DIR)/libdarknet.so

BIN_DIR				=		./bin

BIN_TARGET 			= 		$(BIN_DIR)/$(TARGET)

# COMPILE
CPP					= 		g++
CPPFLAGS			= 		-I $(INCLUDE_DIR)

all : $(BIN_TARGET)

$(BIN_TARGET): $(OBJECT)
	$(CPP) $^ -o $@ $(CFLAGS) $(PKG)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CPP) -c $< -o $@ $(CPPFLAGS)

.PHONY: clean cleanobj

clean:
	-rm $(OBJECT) $(BIN_TARGET)

cleanobj:
	-rm $(OBJECT)