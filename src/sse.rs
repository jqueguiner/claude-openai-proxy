//! Minimal incremental Server-Sent Events parser: feeds bytes in, yields the
//! `data:` payload of each complete event.

#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            let Some(end) = find_event_end(&self.buf) else { break };
            let (block_len, sep_len) = end;
            let block: Vec<u8> = self.buf.drain(..block_len + sep_len).collect();
            let block = String::from_utf8_lossy(&block[..block_len]);
            let mut data_lines: Vec<&str> = Vec::new();
            for line in block.split('\n') {
                let line = line.strip_suffix('\r').unwrap_or(line);
                if let Some(rest) = line.strip_prefix("data:") {
                    data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
                }
            }
            if !data_lines.is_empty() {
                events.push(data_lines.join("\n"));
            }
        }
        events
    }
}

/// Locate the first blank line separating events. Returns (block_len, separator_len).
fn find_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < buf.len() {
        if buf[i] == b'\n' {
            if buf.get(i + 1) == Some(&b'\n') {
                return Some((i, 2));
            }
            if buf.get(i + 1) == Some(&b'\r') && buf.get(i + 2) == Some(&b'\n') {
                return Some((i, 3));
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_events_across_chunks() {
        let mut p = SseParser::default();
        assert!(p.push(b"data: {\"a\":1}\n").is_empty());
        let ev = p.push(b"\ndata: [DONE]\n\n");
        assert_eq!(ev, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
    }

    #[test]
    fn handles_crlf_and_comments() {
        let mut p = SseParser::default();
        let ev = p.push(b": keepalive\r\n\r\nevent: x\r\ndata: 1\r\ndata: 2\r\n\r\n");
        assert_eq!(ev, vec!["1\n2".to_string()]);
    }
}
