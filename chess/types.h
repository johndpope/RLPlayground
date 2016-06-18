#pragma once

#include <array>
#include <vector>
#include <utility>
#include <stdlib.h>

namespace chess {

enum Type {
  NONE,
  KING,
  QUEEN,
  ROOK,
  BISHOP,
  KNIGHT,
  PAWN,
};

enum Color {
  BLACK,
  WHITE,
};

struct Square {
  Type type;
  Color color;
  
  Square() : type(NONE), color(BLACK) {}
  Square(Type type, Color color) : type(type), color(color) {}

  explicit Square(int index) {
    if (index == 0) {
      type = NONE;
      color = BLACK;
    } else if (index < 13) {
      index -= 1;
      type = Type((index >> 1) + 1);
      color = Color(index & 1);
    } else {
      printf("Invalid square.\n");
    }
  }
  
  bool IsEmpty() const {
    return type == NONE;
  }
  
  bool IsFriend(Color color) const {
    return type != NONE && this->color == color;
  }
  
  bool IsEnemy(Color color) const {
    return type != NONE && this->color != color;
  }

  int Index() {
    if (type == NONE) {
      return 0;
    } else {
      return ((type - 1) << 1) + color + 1;
    }
  }
};

struct Vector {
  int row;
  int col;
  
  Vector(): row(0), col(0) {}
  Vector(int row, int col) : row(row), col(col) {}

  explicit Vector(int index) {
    if (index > 63) {
      printf("Invalid vector.\n");
    }
    row = (index >> 3) & 7;
    col = index & 7;
  }
  
  Vector operator+(const Vector& pos) const {
    return Vector(row + pos.row, col + pos.col);
  }

  Vector operator*(int m) const {
    return Vector(row * m, col * m);
  }
  
  bool operator==(Vector pos) const {
    return pos.row == row && pos.col == col;
  }
  
  bool IsValid() const {
    return row >= 0 && row < 8 && col >= 0 && col < 8;
  }

  int Index() const {
    return (row << 3) + col;
  }
};

struct Move {
  Vector start;
  Vector end;

  Move() = default;
  Move(Vector start, Vector end) : start(start), end(end) {}

  explicit Move(int index) {
    if (index > 4095) {
      printf("Invalid vector.\n");
    }
    start = Vector((index >> 6) & 63);
    end = Vector(index & 63);
  }
  
  bool operator==(const Move& move) const {
    return move.start == start && move.end == end;
  }

  int Index() const {
    return (start.Index() << 6) + end.Index();
  }

  bool IsValid() const {
    return start.IsValid() && end.IsValid();
  }
};

struct Board {
  std::array<std::array<Square, 8>, 8> squares;
  
  Square& operator[](const Vector& pos) {
    return squares[pos.row][pos.col];
  }
  
  const Square& operator[](const Vector& pos) const {
    return squares[pos.row][pos.col];
  }

  const Square& At(int index) const {
    Vector pos(index);
    return squares[pos.row][pos.col];
  }

  const std::vector<Square> GetPieces() const {
    std::vector<Square> selected;
    for (int row = 0; row < 8; ++row) {
      for (int col = 0; col < 8; ++col) {
        const auto& square = squares[row][col];
        if (square.type != NONE) {
          selected.push_back(square);
        }
      }
    }
    return selected;
  }

  const std::vector<Square> GetPieces(Color color) const {
    std::vector<Square> selected;
    for (int row = 0; row < 8; ++row) {
      for (int col = 0; col < 8; ++col) {
        const auto& square = squares[row][col];
        if (square.type != NONE && square.color == color) {
          selected.push_back(square);
        }
      }
    }
    return selected;
  }

  const std::vector<Square> GetPieces(Color color, Type type) const {
    std::vector<Square> selected;
    for (int row = 0; row < 8; ++row) {
      for (int col = 0; col < 8; ++col) {
        const auto& square = squares[row][col];
        if (square.type == type && square.color == color) {
          selected.push_back(square);
        }
      }
    }
    return selected;
  }
};

struct State {
  State() {
    Reset();
  }
  State(const State&) = default;

  void Reset() {
    turn = WHITE;
    can_castle[0][0] = true;
    can_castle[0][1] = true;
    can_castle[1][0] = true;
    can_castle[1][1] = true;
    
    Type init[] = { 
      ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK 
    };
    for (int i = 0; i < 8; ++i) {
      board[Vector(0, i)] = Square(init[i], WHITE);
      board[Vector(1, i)] = Square(PAWN, WHITE);
      board[Vector(2, i)] = Square();
      board[Vector(3, i)] = Square();
      board[Vector(4, i)] = Square();
      board[Vector(5, i)] = Square();
      board[Vector(6, i)] = Square(PAWN, BLACK);
      board[Vector(7, i)] = Square(init[i], BLACK);
    }
  }

  bool Play(const Move& move, bool alternate) {  
    if (!move.IsValid()) {
      return false;
    }

    // Perform the move
    board[move.end] = board[move.start];
    board[move.start] = Square();
    
    // Special handling of castling 
    const Square& square = board[move.end];
    if (square.type == KING && 
      abs(move.end.col - move.start.col) > 1 &&
      move.end.row == move.start.row &&
      (move.end.row == 0 || move.end.row == 7)) {
      if (move.end.col > move.start.col) {
        std::swap(
          board[Vector(move.end.row, 7)],
          board[Vector(move.end.row, move.end.col - 1)]);
      } else {
        std::swap(
          board[Vector(move.end.row, 0)],
          board[Vector(move.end.row, move.end.col + 1)]);
      }
    }
    
    // Promote pawn to queen
    if (square.type == PAWN) {
      if ((square.color == WHITE && move.end.row == 7) ||
        (square.color == BLACK && move.end.row == 0)) {
        board[move.end].type = QUEEN;    
      }
    }
    
    // Update can_castle if rook or king moved
    if (square.type == KING) {
      can_castle[turn][0] = false;
      can_castle[turn][1] = false;
    }
    if (square.type == ROOK) {
      if (move.start.col == 0) {
        can_castle[turn][0] = false;
      }
      if (move.start.col == 7) {
        can_castle[turn][1] = false;
      }
    }
    
    if (alternate) {
      turn = Color(1 - turn);
    }

    return true;
  }
  
  Color turn;
  Board board;
  std::array<std::array<bool, 2>, 2> can_castle;
};

}  // namespace chess
