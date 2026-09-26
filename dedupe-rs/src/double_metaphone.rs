//! Double Metaphone, ported from the original C++ implementation shipped with
//! the `doublemetaphone` Python package (`double_metaphone.cc`).
//!
//! The implementation works on UTF-8 bytes to match the C++ `char` handling,
//! including its (deliberately preserved) quirks: only ASCII letters are
//! upper-cased and the output is truncated to 32 characters.

const MAX_LENGTH: isize = 32;

fn make_upper(s: &mut [u8]) {
    for b in s.iter_mut() {
        if b.is_ascii_lowercase() {
            *b = b.to_ascii_uppercase();
        }
    }
}

#[inline]
fn is_vowel(s: &[u8], pos: isize) -> bool {
    if pos < 0 || pos as usize >= s.len() {
        return false;
    }
    matches!(s[pos as usize], b'A' | b'E' | b'I' | b'O' | b'U' | b'Y')
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn slavo_germanic(s: &[u8]) -> bool {
    contains(s, b"W") || contains(s, b"K") || contains(s, b"CZ") || contains(s, b"WITZ")
}

#[inline]
fn get_at(s: &[u8], pos: isize) -> u8 {
    if pos < 0 || pos as usize >= s.len() {
        0
    } else {
        s[pos as usize]
    }
}

/// `strncmp(a, b, n) == 0` semantics, stopping at NUL in either operand.
fn strncmp_eq(a: &[u8], b: &[u8], n: usize) -> bool {
    for k in 0..n {
        let ca = a.get(k).copied().unwrap_or(0);
        let cb = b.get(k).copied().unwrap_or(0);
        if ca != cb {
            return false;
        }
        if ca == 0 {
            return true;
        }
    }
    true
}

/// Faithful port of the C++ `StringAt` helper.
fn string_at(s: &[u8], start: isize, length: usize, tests: &[&str]) -> bool {
    if start < 0 || start as usize >= s.len() {
        return false;
    }
    for t in tests {
        if t.is_empty() {
            break;
        }
        if strncmp_eq(&s[start as usize..], t.as_bytes(), length) {
            return true;
        }
    }
    false
}

/// Return the `(primary, secondary)` Double Metaphone codes for `input`.
pub fn double_metaphone_codes(input: &str) -> (String, String) {
    let length = input.len() as isize;
    let last = length - 1;

    let mut original: Vec<u8> = input.as_bytes().to_vec();
    original.extend_from_slice(b"     ");
    make_upper(&mut original);

    let mut primary: Vec<u8> = Vec::new();
    let mut secondary: Vec<u8> = Vec::new();

    let mut current: isize = 0;

    if string_at(&original, 0, 2, &["GN", "KN", "PN", "WR", "PS", ""]) {
        current += 1;
    }

    if get_at(&original, 0) == b'X' {
        primary.push(b'S');
        secondary.push(b'S');
        current += 1;
    }

    while (primary.len() as isize) < MAX_LENGTH || (secondary.len() as isize) < MAX_LENGTH {
        if current >= length {
            break;
        }

        match get_at(&original, current) {
            b'A' | b'E' | b'I' | b'O' | b'U' | b'Y' => {
                if current == 0 {
                    primary.push(b'A');
                    secondary.push(b'A');
                }
                current += 1;
            }
            b'B' => {
                primary.push(b'P');
                secondary.push(b'P');
                if get_at(&original, current + 1) == b'B' {
                    current += 2;
                } else {
                    current += 1;
                }
            }
            // 'Ç'
            0xC7 => {
                primary.push(b'S');
                secondary.push(b'S');
                current += 1;
            }
            b'C' => {
                if current > 1
                    && !is_vowel(&original, current - 2)
                    && string_at(&original, current - 1, 3, &["ACH", ""])
                    && (get_at(&original, current + 2) != b'I')
                    && ((get_at(&original, current + 2) != b'E')
                        || string_at(&original, current - 2, 6, &["BACHER", "MACHER", ""]))
                {
                    primary.extend_from_slice(b"K");
                    secondary.extend_from_slice(b"K");
                    current += 2;
                } else if current == 0 && string_at(&original, current, 6, &["CAESAR", ""]) {
                    primary.extend_from_slice(b"S");
                    secondary.extend_from_slice(b"S");
                    current += 2;
                } else if string_at(&original, current, 4, &["CHIA", ""]) {
                    primary.extend_from_slice(b"K");
                    secondary.extend_from_slice(b"K");
                    current += 2;
                } else if string_at(&original, current, 2, &["CH", ""]) {
                    if current > 0 && string_at(&original, current, 4, &["CHAE", ""]) {
                        primary.extend_from_slice(b"K");
                        secondary.extend_from_slice(b"X");
                        current += 2;
                    } else if current == 0
                        && (string_at(&original, current + 1, 5, &["HARAC", "HARIS", ""])
                            || string_at(
                                &original,
                                current + 1,
                                3,
                                &["HOR", "HYM", "HIA", "HEM", ""],
                            ))
                        && !string_at(&original, 0, 5, &["CHORE", ""])
                    {
                        primary.extend_from_slice(b"K");
                        secondary.extend_from_slice(b"K");
                        current += 2;
                    } else {
                        if string_at(&original, 0, 4, &["VAN ", "VON ", ""])
                            || string_at(&original, 0, 3, &["SCH", ""])
                            || string_at(
                                &original,
                                current - 2,
                                6,
                                &["ORCHES", "ARCHIT", "ORCHID", ""],
                            )
                            || string_at(&original, current + 2, 1, &["T", "S", ""])
                            || ((string_at(
                                &original,
                                current - 1,
                                1,
                                &["A", "O", "U", "E", ""],
                            ) || current == 0)
                                && string_at(
                                    &original,
                                    current + 2,
                                    1,
                                    &[
                                        "L", "R", "N", "M", "B", "H", "F", "V", "W", " ", "",
                                    ],
                                ))
                        {
                            primary.extend_from_slice(b"K");
                            secondary.extend_from_slice(b"K");
                        } else if current > 0 {
                            if string_at(&original, 0, 2, &["MC", ""]) {
                                primary.extend_from_slice(b"K");
                                secondary.extend_from_slice(b"K");
                            } else {
                                primary.extend_from_slice(b"X");
                                secondary.extend_from_slice(b"K");
                            }
                        } else {
                            primary.extend_from_slice(b"X");
                            secondary.extend_from_slice(b"X");
                        }
                        current += 2;
                    }
                } else if string_at(&original, current, 2, &["CZ", ""])
                    && !string_at(&original, current - 2, 4, &["WICZ", ""])
                {
                    primary.extend_from_slice(b"S");
                    secondary.extend_from_slice(b"X");
                    current += 2;
                } else if string_at(&original, current + 1, 3, &["CIA", ""]) {
                    primary.extend_from_slice(b"X");
                    secondary.extend_from_slice(b"X");
                    current += 3;
                } else if string_at(&original, current, 2, &["CC", ""])
                    && !((current == 1) && (get_at(&original, 0) == b'M'))
                {
                    if string_at(&original, current + 2, 1, &["I", "E", "H", ""])
                        && !string_at(&original, current + 2, 2, &["HU", ""])
                    {
                        if ((current == 1) && (get_at(&original, current - 1) == b'A'))
                            || string_at(&original, current - 1, 5, &["UCCEE", "UCCES", ""])
                        {
                            primary.extend_from_slice(b"KS");
                            secondary.extend_from_slice(b"KS");
                        } else {
                            primary.extend_from_slice(b"X");
                            secondary.extend_from_slice(b"X");
                        }
                        current += 3;
                    } else {
                        primary.extend_from_slice(b"K");
                        secondary.extend_from_slice(b"K");
                        current += 2;
                    }
                } else if string_at(&original, current, 2, &["CK", "CG", "CQ", ""]) {
                    primary.extend_from_slice(b"K");
                    secondary.extend_from_slice(b"K");
                    current += 2;
                } else if string_at(&original, current, 2, &["CI", "CE", "CY", ""]) {
                    if string_at(&original, current, 3, &["CIO", "CIE", "CIA", ""]) {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"X");
                    } else {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"S");
                    }
                    current += 2;
                } else {
                    primary.extend_from_slice(b"K");
                    secondary.extend_from_slice(b"K");
                    if string_at(&original, current + 1, 2, &[" C", " Q", " G", ""]) {
                        current += 3;
                    } else if string_at(&original, current + 1, 1, &["C", "K", "Q", ""])
                        && !string_at(&original, current + 1, 2, &["CE", "CI", ""])
                    {
                        current += 2;
                    } else {
                        current += 1;
                    }
                }
            }
            b'D' => {
                if string_at(&original, current, 2, &["DG", ""]) {
                    if string_at(&original, current + 2, 1, &["I", "E", "Y", ""]) {
                        primary.extend_from_slice(b"J");
                        secondary.extend_from_slice(b"J");
                        current += 3;
                    } else {
                        primary.extend_from_slice(b"TK");
                        secondary.extend_from_slice(b"TK");
                        current += 2;
                    }
                } else if string_at(&original, current, 2, &["DT", "DD", ""]) {
                    primary.extend_from_slice(b"T");
                    secondary.extend_from_slice(b"T");
                    current += 2;
                } else {
                    primary.extend_from_slice(b"T");
                    secondary.extend_from_slice(b"T");
                    current += 1;
                }
            }
            b'F' => {
                if get_at(&original, current + 1) == b'F' {
                    current += 2;
                } else {
                    current += 1;
                }
                primary.extend_from_slice(b"F");
                secondary.extend_from_slice(b"F");
            }
            b'G' => {
                let mut handled = false;
                if get_at(&original, current + 1) == b'H' {
                    if (current > 0) && !is_vowel(&original, current - 1) {
                        primary.extend_from_slice(b"K");
                        secondary.extend_from_slice(b"K");
                        current += 2;
                        handled = true;
                    } else if current == 0 {
                        if get_at(&original, current + 2) == b'I' {
                            primary.extend_from_slice(b"J");
                            secondary.extend_from_slice(b"J");
                        } else {
                            primary.extend_from_slice(b"K");
                            secondary.extend_from_slice(b"K");
                        }
                        current += 2;
                        handled = true;
                    } else if ((current > 1)
                        && string_at(&original, current - 2, 1, &["B", "H", "D", ""]))
                        || ((current > 2)
                            && string_at(&original, current - 3, 1, &["B", "H", "D", ""]))
                        || ((current > 3)
                            && string_at(&original, current - 4, 1, &["B", "H", ""]))
                    {
                        current += 2;
                        handled = true;
                    } else {
                        if (current > 2)
                            && (get_at(&original, current - 1) == b'U')
                            && string_at(
                                &original,
                                current - 3,
                                1,
                                &["C", "G", "L", "R", "T", ""],
                            )
                        {
                            primary.extend_from_slice(b"F");
                            secondary.extend_from_slice(b"F");
                        } else if (current > 0) && get_at(&original, current - 1) != b'I' {
                            primary.extend_from_slice(b"K");
                            secondary.extend_from_slice(b"K");
                        }
                        current += 2;
                        handled = true;
                    }
                }

                if !handled {
                    if get_at(&original, current + 1) == b'N' {
                        if (current == 1) && is_vowel(&original, 0) && !slavo_germanic(&original) {
                            primary.extend_from_slice(b"KN");
                            secondary.extend_from_slice(b"N");
                        } else if !string_at(&original, current + 2, 2, &["EY", ""])
                            && (get_at(&original, current + 1) != b'Y')
                            && !slavo_germanic(&original)
                        {
                            primary.extend_from_slice(b"N");
                            secondary.extend_from_slice(b"KN");
                        } else {
                            primary.extend_from_slice(b"KN");
                            secondary.extend_from_slice(b"KN");
                        }
                        current += 2;
                    } else if string_at(&original, current + 1, 2, &["LI", ""])
                        && !slavo_germanic(&original)
                    {
                        primary.extend_from_slice(b"KL");
                        secondary.extend_from_slice(b"L");
                        current += 2;
                    } else if (current == 0)
                        && ((get_at(&original, current + 1) == b'Y')
                            || string_at(
                                &original,
                                current + 1,
                                2,
                                &[
                                    "ES", "EP", "EB", "EL", "EY", "IB", "IL", "IN",
                                    "IE", "EI", "ER", "",
                                ],
                            ))
                    {
                        primary.extend_from_slice(b"K");
                        secondary.extend_from_slice(b"J");
                        current += 2;
                    } else if (string_at(&original, current + 1, 2, &["ER", ""])
                        || (get_at(&original, current + 1) == b'Y'))
                        && !string_at(&original, 0, 6, &["DANGER", "RANGER", "MANGER", ""])
                        && !string_at(&original, current - 1, 1, &["E", "I", ""])
                        && !string_at(&original, current - 1, 3, &["RGY", "OGY", ""])
                    {
                        primary.extend_from_slice(b"K");
                        secondary.extend_from_slice(b"J");
                        current += 2;
                    } else if string_at(&original, current + 1, 1, &["E", "I", "Y", ""])
                        || string_at(&original, current - 1, 4, &["AGGI", "OGGI", ""])
                    {
                        if string_at(&original, 0, 4, &["VAN ", "VON ", ""])
                            || string_at(&original, 0, 3, &["SCH", ""])
                            || string_at(&original, current + 1, 2, &["ET", ""])
                        {
                            primary.extend_from_slice(b"K");
                            secondary.extend_from_slice(b"K");
                        } else if string_at(&original, current + 1, 4, &["IER ", ""]) {
                            primary.extend_from_slice(b"J");
                            secondary.extend_from_slice(b"J");
                        } else {
                            primary.extend_from_slice(b"J");
                            secondary.extend_from_slice(b"K");
                        }
                        current += 2;
                    } else {
                        if get_at(&original, current + 1) == b'G' {
                            current += 2;
                        } else {
                            current += 1;
                        }
                        primary.extend_from_slice(b"K");
                        secondary.extend_from_slice(b"K");
                    }
                }
            }
            b'H' => {
                if ((current == 0) || is_vowel(&original, current - 1))
                    && is_vowel(&original, current + 1)
                {
                    primary.extend_from_slice(b"H");
                    secondary.extend_from_slice(b"H");
                    current += 2;
                } else {
                    current += 1;
                }
            }
            b'J' => {
                if string_at(&original, current, 4, &["JOSE", ""])
                    || string_at(&original, 0, 4, &["SAN ", ""])
                {
                    if ((current == 0) && (get_at(&original, current + 4) == b' '))
                        || string_at(&original, 0, 4, &["SAN ", ""])
                    {
                        primary.extend_from_slice(b"H");
                        secondary.extend_from_slice(b"H");
                    } else {
                        primary.extend_from_slice(b"J");
                        secondary.extend_from_slice(b"H");
                    }
                    current += 1;
                } else {
                    if (current == 0) && !string_at(&original, current, 4, &["JOSE", ""]) {
                        primary.extend_from_slice(b"J");
                        secondary.extend_from_slice(b"A");
                    } else if is_vowel(&original, current - 1)
                        && !slavo_germanic(&original)
                        && ((get_at(&original, current + 1) == b'A')
                            || (get_at(&original, current + 1) == b'O'))
                    {
                        primary.extend_from_slice(b"J");
                        secondary.extend_from_slice(b"H");
                    } else if current == last {
                        primary.extend_from_slice(b"J");
                    } else if !string_at(
                        &original,
                        current + 1,
                        1,
                        &["L", "T", "K", "S", "N", "M", "B", "Z", ""],
                    ) && !string_at(&original, current - 1, 1, &["S", "K", "L", ""])
                    {
                        primary.extend_from_slice(b"J");
                        secondary.extend_from_slice(b"J");
                    }

                    if get_at(&original, current + 1) == b'J' {
                        current += 2;
                    } else {
                        current += 1;
                    }
                }
            }
            b'K' => {
                if get_at(&original, current + 1) == b'K' {
                    current += 2;
                } else {
                    current += 1;
                }
                primary.extend_from_slice(b"K");
                secondary.extend_from_slice(b"K");
            }
            b'L' => {
                if get_at(&original, current + 1) == b'L' {
                    if ((current == (length - 3))
                        && string_at(
                            &original,
                            current - 1,
                            4,
                            &["ILLO", "ILLA", "ALLE", ""],
                        ))
                        || ((string_at(&original, last - 1, 2, &["AS", "OS", ""])
                            || string_at(&original, last, 1, &["A", "O", ""]))
                            && string_at(&original, current - 1, 4, &["ALLE", ""]))
                    {
                        primary.extend_from_slice(b"L");
                        current += 2;
                    } else {
                        current += 2;
                        primary.extend_from_slice(b"L");
                        secondary.extend_from_slice(b"L");
                    }
                } else {
                    current += 1;
                    primary.extend_from_slice(b"L");
                    secondary.extend_from_slice(b"L");
                }
            }
            b'M' => {
                if (string_at(&original, current - 1, 3, &["UMB", ""])
                    && (((current + 1) == last)
                        || string_at(&original, current + 2, 2, &["ER", ""])))
                    || (get_at(&original, current + 1) == b'M')
                {
                    current += 2;
                } else {
                    current += 1;
                }
                primary.extend_from_slice(b"M");
                secondary.extend_from_slice(b"M");
            }
            b'N' => {
                if get_at(&original, current + 1) == b'N' {
                    current += 2;
                } else {
                    current += 1;
                }
                primary.extend_from_slice(b"N");
                secondary.extend_from_slice(b"N");
            }
            // 'Ñ'
            0xD1 => {
                current += 1;
                primary.extend_from_slice(b"N");
                secondary.extend_from_slice(b"N");
            }
            b'P' => {
                if get_at(&original, current + 1) == b'H' {
                    primary.extend_from_slice(b"F");
                    secondary.extend_from_slice(b"F");
                    current += 2;
                } else {
                    if string_at(&original, current + 1, 1, &["P", "B", ""]) {
                        current += 2;
                    } else {
                        current += 1;
                    }
                    primary.extend_from_slice(b"P");
                    secondary.extend_from_slice(b"P");
                }
            }
            b'Q' => {
                if get_at(&original, current + 1) == b'Q' {
                    current += 2;
                } else {
                    current += 1;
                }
                primary.extend_from_slice(b"K");
                secondary.extend_from_slice(b"K");
            }
            b'R' => {
                if (current == last)
                    && !slavo_germanic(&original)
                    && string_at(&original, current - 2, 2, &["IE", ""])
                    && !string_at(&original, current - 4, 2, &["ME", "MA", ""])
                {
                    secondary.extend_from_slice(b"R");
                } else {
                    primary.extend_from_slice(b"R");
                    secondary.extend_from_slice(b"R");
                }
                if get_at(&original, current + 1) == b'R' {
                    current += 2;
                } else {
                    current += 1;
                }
            }
            b'S' => {
                if string_at(&original, current - 1, 3, &["ISL", "YSL", ""]) {
                    current += 1;
                } else if (current == 0) && string_at(&original, current, 5, &["SUGAR", ""]) {
                    primary.extend_from_slice(b"X");
                    secondary.extend_from_slice(b"S");
                    current += 1;
                } else if string_at(&original, current, 2, &["SH", ""]) {
                    if string_at(
                        &original,
                        current + 1,
                        4,
                        &["HEIM", "HOEK", "HOLM", "HOLZ", ""],
                    ) {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"S");
                    } else {
                        primary.extend_from_slice(b"X");
                        secondary.extend_from_slice(b"X");
                    }
                    current += 2;
                } else if string_at(&original, current, 3, &["SIO", "SIA", ""])
                    || string_at(&original, current, 4, &["SIAN", ""])
                {
                    if !slavo_germanic(&original) {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"X");
                    } else {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"S");
                    }
                    current += 3;
                } else if ((current == 0)
                    && string_at(&original, current + 1, 1, &["M", "N", "L", "W", ""]))
                    || string_at(&original, current + 1, 1, &["Z", ""])
                {
                    primary.extend_from_slice(b"S");
                    secondary.extend_from_slice(b"X");
                    if string_at(&original, current + 1, 1, &["Z", ""]) {
                        current += 2;
                    } else {
                        current += 1;
                    }
                } else if string_at(&original, current, 2, &["SC", ""]) {
                    if get_at(&original, current + 2) == b'H' {
                        if string_at(
                            &original,
                            current + 3,
                            2,
                            &["OO", "ER", "EN", "UY", "ED", "EM", ""],
                        ) {
                            if string_at(&original, current + 3, 2, &["ER", "EN", ""]) {
                                primary.extend_from_slice(b"X");
                                secondary.extend_from_slice(b"SK");
                            } else {
                                primary.extend_from_slice(b"SK");
                                secondary.extend_from_slice(b"SK");
                            }
                            current += 3;
                        } else {
                            if (current == 0)
                                && !is_vowel(&original, 3)
                                && (get_at(&original, 3) != b'W')
                            {
                                primary.extend_from_slice(b"X");
                                secondary.extend_from_slice(b"S");
                            } else {
                                primary.extend_from_slice(b"X");
                                secondary.extend_from_slice(b"X");
                            }
                            current += 3;
                        }
                    } else if string_at(&original, current + 2, 1, &["I", "E", "Y", ""]) {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"S");
                        current += 3;
                    } else {
                        primary.extend_from_slice(b"SK");
                        secondary.extend_from_slice(b"SK");
                        current += 3;
                    }
                } else {
                    if (current == last)
                        && string_at(&original, current - 2, 2, &["AI", "OI", ""])
                    {
                        secondary.extend_from_slice(b"S");
                    } else {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"S");
                    }

                    if string_at(&original, current + 1, 1, &["S", "Z", ""]) {
                        current += 2;
                    } else {
                        current += 1;
                    }
                }
            }
            b'T' => {
                if string_at(&original, current, 4, &["TION", ""]) {
                    primary.extend_from_slice(b"X");
                    secondary.extend_from_slice(b"X");
                    current += 3;
                } else if string_at(&original, current, 3, &["TIA", "TCH", ""]) {
                    primary.extend_from_slice(b"X");
                    secondary.extend_from_slice(b"X");
                    current += 3;
                } else if string_at(&original, current, 2, &["TH", ""])
                    || string_at(&original, current, 3, &["TTH", ""])
                {
                    if string_at(&original, current + 2, 2, &["OM", "AM", ""])
                        || string_at(&original, 0, 4, &["VAN ", "VON ", ""])
                        || string_at(&original, 0, 3, &["SCH", ""])
                    {
                        primary.extend_from_slice(b"T");
                        secondary.extend_from_slice(b"T");
                    } else {
                        primary.extend_from_slice(b"0");
                        secondary.extend_from_slice(b"T");
                    }
                    current += 2;
                } else {
                    if string_at(&original, current + 1, 1, &["T", "D", ""]) {
                        current += 2;
                    } else {
                        current += 1;
                    }
                    primary.extend_from_slice(b"T");
                    secondary.extend_from_slice(b"T");
                }
            }
            b'V' => {
                if get_at(&original, current + 1) == b'V' {
                    current += 2;
                } else {
                    current += 1;
                }
                primary.extend_from_slice(b"F");
                secondary.extend_from_slice(b"F");
            }
            b'W' => {
                if string_at(&original, current, 2, &["WR", ""]) {
                    primary.extend_from_slice(b"R");
                    secondary.extend_from_slice(b"R");
                    current += 2;
                } else {
                    if (current == 0)
                        && (is_vowel(&original, current + 1)
                            || string_at(&original, current, 2, &["WH", ""]))
                    {
                        if is_vowel(&original, current + 1) {
                            primary.extend_from_slice(b"A");
                            secondary.extend_from_slice(b"F");
                        } else {
                            primary.extend_from_slice(b"A");
                            secondary.extend_from_slice(b"A");
                        }
                    }

                    if ((current == last) && is_vowel(&original, current - 1))
                        || string_at(
                            &original,
                            current - 1,
                            5,
                            &["EWSKI", "EWSKY", "OWSKI", "OWSKY", ""],
                        )
                        || string_at(&original, 0, 3, &["SCH", ""])
                    {
                        secondary.extend_from_slice(b"F");
                        current += 1;
                    } else if string_at(&original, current, 4, &["WICZ", "WITZ", ""]) {
                        primary.extend_from_slice(b"TS");
                        secondary.extend_from_slice(b"FX");
                        current += 4;
                    } else {
                        current += 1;
                    }
                }
            }
            b'X' => {
                if !((current == last)
                    && (string_at(&original, current - 3, 3, &["IAU", "EAU", ""])
                        || string_at(&original, current - 2, 2, &["AU", "OU", ""])))
                {
                    primary.extend_from_slice(b"KS");
                    secondary.extend_from_slice(b"KS");
                }

                if string_at(&original, current + 1, 1, &["C", "X", ""]) {
                    current += 2;
                } else {
                    current += 1;
                }
            }
            b'Z' => {
                if get_at(&original, current + 1) == b'H' {
                    primary.extend_from_slice(b"J");
                    secondary.extend_from_slice(b"J");
                    current += 2;
                } else {
                    if string_at(&original, current + 1, 2, &["ZO", "ZI", "ZA", ""])
                        || (slavo_germanic(&original)
                            && ((current > 0) && get_at(&original, current - 1) != b'T'))
                    {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"TS");
                    } else {
                        primary.extend_from_slice(b"S");
                        secondary.extend_from_slice(b"S");
                    }

                    if get_at(&original, current + 1) == b'Z' {
                        current += 2;
                    } else {
                        current += 1;
                    }
                }
            }
            _ => {
                current += 1;
            }
        }
    }

    if primary.len() as isize > MAX_LENGTH {
        primary.truncate(MAX_LENGTH as usize);
    }
    if secondary.len() as isize > MAX_LENGTH {
        secondary.truncate(MAX_LENGTH as usize);
    }

    (
        String::from_utf8_lossy(&primary).into_owned(),
        String::from_utf8_lossy(&secondary).into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codes(s: &str) -> (String, String) {
        double_metaphone_codes(s)
    }

    #[test]
    fn basic() {
        assert_eq!(codes("i"), ("A".into(), "A".into()));
        assert_eq!(codes("donald"), ("TNLT".into(), "TNLT".into()));
        assert_eq!(codes("goofy"), ("KF".into(), "KF".into()));
        assert_eq!(codes("cipciop"), ("SPSP".into(), "SPXP".into()));
        assert_eq!(codes("smith"), ("SM0".into(), "XMT".into()));
        assert_eq!(codes("schmidt"), ("XMT".into(), "SMT".into()));
    }

    #[test]
    fn long_truncated_to_32() {
        let s = "a".repeat(100);
        let (p, q) = codes(&s);
        assert!(p.len() <= 32 && q.len() <= 32);
    }
}
