#include <string>
#include <vector>

#include "game.h"
#include "types.h"

namespace chess {

const wchar_t* SquareToString(const Square& square);  
const wchar_t* SquareToString(const Square& square, Vector pos);
std::string VectorToString(const Vector& vec);
std::string MoveToString(const Move& move);
std::wstring BoardToString(const Board& board);

std::vector<Move> Parse(const State& state, const char* str);

float GetStateValue(const Game& game);
std::vector<float> GetActionValues(
    const Game& game, const std::vector<Move>& moves, int depth);

}  // namespace chess
