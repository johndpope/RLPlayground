%module chess

%include <std_vector.i>
%include <std_string.i>
%include <std_wstring.i>

%{
#include "types.h"
#include "game.h"
#include "utils.h"
%}

namespace std {
  %template(FloatVector) std::vector<float>;
  %template(MoveVector) std::vector<chess::Move>;
  %template(SquareVector) std::vector<chess::Square>;
}

%include "types.h"
%include "game.h"
%include "utils.h"

%extend chess::Square
{
  std::wstring __unicode__() {
    return chess::SquareToString(*self);
  }
}

%extend chess::Vector
{
  std::string __str__() {
    return chess::VectorToString(*self);
  }
};

%extend chess::Move
{
  std::string __str__() {
    return chess::MoveToString(*self);
  }
};

%extend chess::Board
{  
  chess::Square& __getitem__(chess::Vector pos) {
    return self->squares[pos.row][pos.col];
  }
  
  void __setitem__(chess::Vector pos, const chess::Square& square) {
    self->squares[pos.row][pos.col] = square;
  }
  
  std::wstring __unicode__() {
    return chess::BoardToString(*self);
  }
};
