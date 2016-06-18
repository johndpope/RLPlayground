#include <stdio.h>
#include <string.h>
#include <sstream>

#include "io.h" 

namespace chess {

const wchar_t* SquareToString(const Square& square) {
  static const wchar_t* piece_code[] = {
    L"♔", L"♕", L"♖", L"♗", L"♘", L"♙",
    L"♚", L"♛", L"♜", L"♝", L"♞", L"♟"};
  if (square.type == NONE) {
    return L" ";
  }
  return piece_code[square.type - 1 + square.color * 6];
}

const wchar_t* SquareToString(const Square& square, Vector pos) {
  static const wchar_t* block_code[] = {
    L"◾", L"◽"
  };
  if (square.type == NONE) {
    return block_code[(pos.row + pos.col) % 2];
  }
  return SquareToString(square);
}

std::string VectorToString(const Vector& vec) {
  std::stringstream ss;
  ss << "(" << vec.row << "," << vec.col << ")";
  return ss.str();
}

std::string MoveToString(const Move& move) {
  std::stringstream ss;
  ss << "(" << move.start.row << "," << move.start.col << ") -> ("  
     << move.end.row << "," << move.end.col << ")";
  return ss.str();
}

std::wstring BoardToString(const Board& board) {
  std::wstringstream ss;
  for (int i = 0; i < 8; ++i) {
    ss << " " << char(i + 'a');
  }
  ss << std::endl << " ";
  for (int i = 0; i < 15; ++i) {
    ss << "-";
  }
  ss << std::endl;
  for (int j = 0; j < 8; ++j) {
    ss << 8 - j;
    for (int i = 0; i < 8; ++i) {
      Vector pos(7 - j, i);
      const Square& square = board[pos];
      ss << SquareToString(square, pos) << " ";
    }
    ss << 8 - j << std::endl;
  }
  ss << " ";
  for (int i = 0; i < 15; ++i) {
    ss << "-";
  }
  ss << std::endl;
  for (int i = 0; i < 8; ++i) {
    ss << " " << char(i + 'a');
  }
  return ss.str();
}

std::vector<Move> Parse(const State& state, const char* str) {
  auto len = strlen(str);
  std::vector<Move> moves;
  if (strcmp(str, "oo") == 0) {
    // Short castle
    if (state.board[Vector(0, 4)].type == KING) {
      moves.emplace_back(Vector(0, 4), Vector(0, 6));
    }
    if (state.board[Vector(7, 4)].type == KING) {
      moves.emplace_back(Vector(7, 4), Vector(7, 6));
    }
  } else if (strcmp(str, "ooo") == 0) {
    // Long castle
    if (state.board[Vector(0, 4)].type == KING) {
      moves.emplace_back(Vector(0, 4), Vector(0, 2));
    }
    if (state.board[Vector(7, 4)].type == KING) {
      moves.emplace_back(Vector(7, 4), Vector(7, 2));
    }
  } else if (len == 2 || len == 4) {
    Vector pos(str[1] - '1', str[0] - 'a');
    if (!pos.IsValid())
      return moves;
    
    if (len == 2) {
      // Pawn move
      if (pos.row > 1 && 
        state.board[Vector(pos.row - 1, pos.col)].type == PAWN) {
        moves.emplace_back(Vector(pos.row - 1, pos.col), pos);
      }
      if (pos.row == 3 && 
        state.board[Vector(pos.row - 2, pos.col)].type == PAWN) {
        moves.emplace_back(Vector(pos.row - 2, pos.col), pos);
      }
      if (pos.row < 6 && 
        state.board[Vector(pos.row + 1, pos.col)].type == PAWN) {
        moves.emplace_back(Vector(pos.row + 1, pos.col), pos);
      }
      if (pos.row == 4 && 
        state.board[Vector(pos.row + 2, pos.col)].type == PAWN) {
        moves.emplace_back(Vector(pos.row + 2, pos.col), pos);
      }
    } else if (len == 4) {
      // Regular move
      Vector end(str[3] - '1', str[2] - 'a');
      moves.emplace_back(pos, end);
    }
  }
  return moves;
}

}  // namespace chess
