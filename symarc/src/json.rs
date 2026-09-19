//! Minimal JSON reader, enough for ARC task files. No dependencies.

#[allow(dead_code)] // The parser accepts JSON values beyond the numeric arrays used by ARC.
#[derive(Debug, Clone)]
pub enum Json {
    Num(i64),
    Str(String),
    Bool(bool),
    Null,
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => panic!("expected array"),
        }
    }
    pub fn num(&self) -> i64 {
        match self {
            Json::Num(n) => *n,
            _ => panic!("expected number"),
        }
    }
}

pub fn parse(s: &[u8]) -> Json {
    let mut p = Parser { s, i: 0 };
    p.ws();
    let v = p.value();
    v
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Json {
        match self.s[self.i] {
            b'{' => self.obj(),
            b'[' => self.array(),
            b'"' => Json::Str(self.string()),
            b't' => {
                self.i += 4;
                Json::Bool(true)
            }
            b'f' => {
                self.i += 5;
                Json::Bool(false)
            }
            b'n' => {
                self.i += 4;
                Json::Null
            }
            _ => self.number(),
        }
    }
    fn obj(&mut self) -> Json {
        self.i += 1; // {
        let mut out = Vec::new();
        loop {
            self.ws();
            if self.s[self.i] == b'}' {
                self.i += 1;
                return Json::Obj(out);
            }
            let k = self.string();
            self.ws();
            self.i += 1; // :
            self.ws();
            let v = self.value();
            out.push((k, v));
            self.ws();
            if self.s[self.i] == b',' {
                self.i += 1;
            }
        }
    }
    fn array(&mut self) -> Json {
        self.i += 1; // [
        let mut out = Vec::new();
        loop {
            self.ws();
            if self.s[self.i] == b']' {
                self.i += 1;
                return Json::Arr(out);
            }
            out.push(self.value());
            self.ws();
            if self.s[self.i] == b',' {
                self.i += 1;
            }
        }
    }
    fn string(&mut self) -> String {
        self.i += 1; // "
        let start = self.i;
        while self.s[self.i] != b'"' {
            self.i += 1;
        }
        let s = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
        self.i += 1;
        s
    }
    fn number(&mut self) -> Json {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
            self.i += 1;
        }
        let n: i64 = std::str::from_utf8(&self.s[start..self.i])
            .unwrap()
            .parse()
            .unwrap();
        Json::Num(n)
    }
}
