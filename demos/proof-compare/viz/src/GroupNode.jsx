import { Handle, Position } from "@xyflow/react";
import { fmtFlops } from "./graph-model.js";

// Input annotation for a collapsed run: total tokens ingested, plus the
// context growth across the run (e.g. a 689-decode stream that started at
// ctx 604 and ended at ctx 1292). Compact — no separators in the range, or
// "ctx 864→1,212" truncates at node width and loses the end value.
function groupInput(data) {
  const tok = `${(data.tokens || 0).toLocaleString()} tok`;
  const a = data.ctxFirst || 0;
  const b = data.ctxLast || 0;
  if (b > a) return `${tok} · ctx ${a}→${b}`;
  if (a > 0) return `${tok} + ${a} ctx`;
  return `in: ${tok}`;
}

// A collapsed run of N atomic forward passes (e.g. "write p_less.py" = 1200
// decode nodes). Shows the summed cost; click to expand into the atomic nodes.
// The underlying data is unchanged — this is purely a display node.
export default function GroupNode({ data }) {
  const tip = `${data.count} × ${data.groupKind} forward passes\n`
    + `${groupInput(data)}\n`
    + `summed cost: ${fmtFlops(data.flops || 0)}\n`
    + `click to expand`;
  return (
    <div className="task-node group-node" style={{ borderColor: data.color }} title={tip}>
      <Handle type="target" position={Position.Top} />
      <div className="task-kind" style={{ color: data.color }}>
        ▸ {data.groupKind} ×{data.count}
      </div>
      <div className="task-label">{(data.label || data.groupKind).slice(0, 28)}</div>
      <div className="task-in">{groupInput(data)}</div>
      <div className="task-flops">{fmtFlops(data.flops || 0)}</div>
      <div className="task-bar-track">
        <div
          className="task-bar-fill"
          style={{ width: `${Math.max(2, (data.barFrac || 0) * 100)}%`, background: data.color }}
        />
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
