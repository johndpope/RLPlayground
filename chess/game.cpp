#include "game.h"

#include <algorithm>
#include <stdlib.h>

namespace chess {
  
const Vector kUp(1, 0);
const Vector kDown(-1, 0);
const Vector kRight(0, 1);
const Vector kLeft(0, -1);

const Vector kUpRight(1, 1);
const Vector kDownRight(-1, 1);
const Vector kUpLeft(1, -1);
const Vector kDownLeft(-1, -1);

const Vector kDoubleUp(2, 0);
const Vector kDoubleDown(-2, 0);
const Vector kDoubleRight(0, 2);
const Vector kDoubleLeft(0, -2);

const std::vector<Vector> kAllDirections({
  kUp, kUpRight, kRight, kDownRight, 
  kDown, kDownLeft, kLeft, kUpLeft});
const std::vector<Vector> kMainDirections({
  kUp, kRight,  kDown, kLeft});
const std::vector<Vector> kCrossDirections({
  kUpRight, kDownRight,  kDownLeft, kUpLeft});
const std::vector<Vector> kLDirections({
  Vector(2,  1), Vector(1,  2), Vector(-1,  2), Vector(-2,  1),
  Vector(2, -1), Vector(1, -2), Vector(-1, -2), Vector(-2, -1)});
const std::vector<Vector> kLeftAndRight({kLeft, kRight});

Game::Game() {
  Update(true);
}

void Game::Reset() {
  state.Reset();
  Update(true);
}

bool Game::Play(const Move& move) {
  if (!IsMoveValid(move)) {
    return false;
  }
  
  state.Play(move, true);
  Update(true);
  return true;
}

void Game::Update(bool update_moves) {
  moves.clear();
  ClearThreats();
  check[0] = false;
  check[1] = false;
  impossible = false;
  
  // Update the threats for the opponent
  Color opponent = Color(1 - state.turn);
  Vector pos;
  for (pos.row = 0; pos.row < 8; ++pos.row) {
    for (pos.col = 0; pos.col < 8; ++pos.col) {
      if (!state.board[pos].IsFriend(opponent)) continue;
      Update(pos, nullptr);
    }
  } 
  
  // Update moves and threats for the player
  std::vector<Vector> positions;
  for (pos.row = 0; pos.row < 8; ++pos.row) {
    for (pos.col = 0; pos.col < 8; ++pos.col) {
      if (!state.board[pos].IsFriend(state.turn)) continue;
      if (update_moves) {
        positions.clear();
        Update(pos, &positions);
        for (auto to: positions) {
          Move move(pos, to);
          if (!WillCheck(move, state.turn)) {
            moves.emplace_back(move);
          }
        }
      } else {
        Update(pos, nullptr);
      }
    }
  } 
  
  // Determine if in check and draw conditions 
  int count = 0;
  int bishop[2] = { 0 };
  int knight = 0;
  for (pos.row = 0; pos.row < 8; ++pos.row) {
    for (pos.col = 0; pos.col < 8; ++pos.col) {
      const auto& square = state.board[pos];
      switch (square.type) {
      case KING:
        check[square.color] = !IsSafe(pos, square.color);
        break;
      case BISHOP:
        ++bishop[(pos.row + pos.col) % 2];
        break;
      case KNIGHT:
        ++knight;
        break;
      default:
        break;
      }
      if (square.type != NONE) {
        ++count;
      }
    }
  }

  impossible = (count == 2 ||
    (count == 3 && (bishop[0] || bishop[1] || knight)) ||
    (count == 4 && (bishop[0] == 2 || bishop[1] == 2)));
}

void Game::Update(Vector pos, std::vector<Vector>* positions) {
  const Square& square = state.board[pos];
  switch (square.type) {
  case KING:
    UpdateKing(pos, positions);
    break;
  case QUEEN:
    UpdateQueen(pos, positions);
    break;
  case ROOK:
    UpdateRook(pos, positions);
    break;
  case BISHOP:
    UpdateBishop(pos, positions);
    break;
  case KNIGHT:
    UpdateKnight(pos, positions);
    break;
  case PAWN:
    UpdatePawn(pos, positions);
    break;
  default:
    break;
  }
}
  
void Game::UpdateKing(Vector pos, std::vector<Vector>* positions) {
  Color color = state.board[pos].color;
  
  for (Vector dir: kAllDirections) {
    Vector to = pos + dir;
    if (!to.IsValid()) continue;
    Threaten(to, color);
    if (positions != nullptr && !state.board[to].IsFriend(color)) {
      positions->push_back(to);
    }
  }
  if (positions != nullptr) {
    const int length[2] = { 3, 2 };
    const Vector dir[2] = { kLeft, kRight };
    for (int j = 0; j < 2; ++j) {
      if (!IsSafe(pos, color) ||
        !IsSafe(pos + dir[j], color) ||
        !IsSafe(pos + dir[j] * 2, color) ||
        !state.can_castle[color][j]) {
        continue;
      }
      const Square& square = state.board[pos + dir[j] * (length[j] + 1)];
      if (square.type != ROOK || square.color != color) {
        continue;
      }
      bool path_clear = true;
      for (int i = 0; i < length[j]; ++i) {
        if (state.board[pos + dir[j] * (i + 1)].type != NONE) {
          path_clear = false;
          break;
        }
      }
      if (!path_clear) {
        continue;
      }
      positions->push_back(pos + dir[j] * 2);
    }
  }
}

void Game::UpdateQueen(Vector pos, std::vector<Vector>* positions) {
  UpdateDirectional(pos, kAllDirections, positions);
}

void Game::UpdateRook(Vector pos, std::vector<Vector>* positions) {
  UpdateDirectional(pos, kMainDirections, positions);
}

void Game::UpdateBishop(Vector pos, std::vector<Vector>* positions) {
  UpdateDirectional(pos, kCrossDirections, positions);
}

void Game::UpdateKnight(Vector pos, std::vector<Vector>* positions) {
  Color color = state.board[pos].color;
  for (Vector dir: kLDirections) {
    Vector to = pos + dir;
    if (!to.IsValid()) continue;
    Threaten(to, color);
    if (positions != nullptr && !state.board[to].IsFriend(color)) {
      positions->push_back(to);
    }
  }
}

void Game::UpdatePawn(Vector pos, std::vector<Vector>* positions) {
  const int begin[2] = { 6, 1 };
  const int end[2] = { 0, 7 };
  const Vector vdir[2] = { kDown, kUp };
  
  Square& square = state.board[pos]; 
  Color color = square.color;
  
  if (pos.row == end[color]) return;
  
  Vector to = pos + vdir[color];
  if (positions != nullptr && state.board[to].IsEmpty()) {
    positions->push_back(to);
    to = pos + (vdir[color] * 2);
    if (pos.row == begin[color] && state.board[to].IsEmpty()) {
      positions->push_back(to);
    }
  }

  for (auto hdir : kLeftAndRight) {
    to = pos + vdir[color] + hdir;
    if (to.IsValid()) {
      Threaten(to, color);
      if (positions != nullptr && state.board[to].IsEnemy(color)) {
        positions->push_back(to);
      }
    }
  }
}

void Game::UpdateDirectional(Vector pos, const std::vector<Vector>& dirs, 
  std::vector<Vector>* positions) {
  Color color = state.board[pos].color;
  for (Vector dir : dirs) {
    Vector to = pos;
    for (int i = 0; i < 7; ++i) {
      to = to + dir;
      if (!to.IsValid()) break;
      Threaten(to, color);
      if (state.board[to].type != NONE) {
        if (state.board[to].color == color) {
          break;
        } else {
          if (positions != nullptr) {
            positions->push_back(to);  
          }
          break;    
        }
      }
      if (positions != nullptr) {
        positions->push_back(to);  
      }
    }
  }
}

bool Game::WillCheck(const Move& move, Color side) const {
  Game game(*this);
  game.state.Play(move, false);
  game.Update(false);
  return game.check[side];
}

void Game::ClearThreats() {
  for (int c = 0; c < 2; ++c) {
    for (int row = 0; row < 8; ++row) {
      for (int col = 0; col < 8; ++col) {
        threats[c][row][col] = false;
      }
    }
  }
}

void Game::Threaten(const Vector& pos, Color color) {
  ++threats[color][pos.row][pos.col];
}

bool Game::IsSafe(const Vector& pos, Color color) const {
  return threats[1 - color][pos.row][pos.col] == 0;
}

bool Game::IsMoveValid(const Move& move) const {
  return std::find(moves.begin(), moves.end(), move) != moves.end();
}

}  // namespace chess
