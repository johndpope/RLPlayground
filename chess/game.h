#pragma once

#include <array>
#include <vector>

#include "types.h"

namespace chess {

class Game {
public:
  Game();
  Game(const Game&) = default;

  void Reset();

  const std::vector<Move>& GetMoves() const { return moves; }
  State& GetState() { return state; }

  bool IsCheck() const { return check[state.turn]; }
  bool IsCheckmate() const { return check[state.turn] && moves.size() == 0; }
  bool IsDraw() const { 
    return impossible || (!check[state.turn] && moves.size() == 0); 
  }
  bool IsEnded() { return IsDraw() || IsCheckmate(); }

  bool Play(const Move& move);

private:
  void UpdateDirectional(Vector pos, const std::vector<Vector>& dirs, 
    std::vector<Vector>* positions);
  void UpdateKing(Vector pos, std::vector<Vector>* positions);
  void UpdateQueen(Vector pos, std::vector<Vector>* positions);
  void UpdateRook(Vector pos, std::vector<Vector>* positions);
  void UpdateBishop(Vector pos, std::vector<Vector>* positions);
  void UpdateKnight(Vector pos, std::vector<Vector>* positions);
  void UpdatePawn(Vector pos, std::vector<Vector>* positions);
  void Update(Vector pos, std::vector<Vector>* positions);
  void Update(bool update_moves);
  bool WillCheck(const Move& move, Color side) const;
  void ClearThreats();
  void Threaten(const Vector& pos, Color color);
  bool IsSafe(const Vector& pos, Color color) const;
  bool IsMoveValid(const Move& move) const;

  State state;
  std::vector<Move> moves;
  std::array<std::array<std::array<int, 8>, 8>, 2> threats;
  std::array<bool, 2> check;
  bool impossible;
};

}  // namespace chess
