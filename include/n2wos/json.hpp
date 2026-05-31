#pragma once

#include <iomanip>
#include <sstream>
#include <string>

namespace n2wos {

inline std::string json_escape(const std::string& input) {
  std::ostringstream out;
  for (char c : input) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(c));
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

inline std::string json_quote(const std::string& input) {
  return std::string("\"") + json_escape(input) + "\"";
}

}  // namespace n2wos
