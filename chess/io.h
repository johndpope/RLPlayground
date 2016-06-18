#include <string>
#include <vector>

#include "types.h"

namespace chess {

const wchar_t* SquareToString(const Square& square);  
const wchar_t* SquareToString(const Square& square, Vector pos);
std::string VectorToString(const Vector& vec);
std::string MoveToString(const Move& move);
std::wstring BoardToString(const Board& board);

std::vector<Move> Parse(const State& state, const char* str);

}  // namespace chess
