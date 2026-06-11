import { describe, it, expect } from "vitest";
import { ctx0, inputSummary } from "./graph-model.js";

// The node annotation "how big is this pass's input" is DERIVED from the
// trace's exact attention accounting, not stored: attended = Σ_{j=1..t}(c0+j).
describe("ctx0", () => {
  it("a prefill has no prior context", () => {
    // p=600: attended = 600*601/2
    expect(ctx0(600, (600 * 601) / 2)).toBe(0);
  });

  it("a decode's context is everything before its token", () => {
    // the pass consuming token #605 attends 605 keys (604 prior + itself)
    expect(ctx0(1, 605)).toBe(604);
  });

  it("a spec verify recovers the committed context", () => {
    // t=5 positions on top of ctx 20: attended = Σ_{j=1..5}(20+j) = 115
    expect(ctx0(5, 115)).toBe(20);
  });

  it("a packed train step has no prior context", () => {
    // batch=4, seq=64: tokens=256, attended = 4 * 64*65/2
    expect(ctx0(256, 4 * (64 * 65) / 2)).toBe(0);
  });

  it("never goes negative and tolerates zeros", () => {
    expect(ctx0(0, 0)).toBe(0);
    expect(ctx0(10, 0)).toBe(0);
  });
});

describe("inputSummary", () => {
  it("prefill: tokens only", () => {
    expect(inputSummary(600, (600 * 601) / 2)).toBe("in: 600 tok");
  });

  it("decode: one token plus its context", () => {
    expect(inputSummary(1, 605)).toBe("in: 1 tok + 604 ctx");
  });

  it("large counts get thousands separators", () => {
    expect(inputSummary(1, 4001)).toBe("in: 1 tok + 4,000 ctx");
  });

  it("empty for a zero-token node", () => {
    expect(inputSummary(0, 0)).toBe("");
  });
});
