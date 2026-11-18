"""The authors' own cost function of a bill-of-materials row (criterion C5), driven by gt/price_table.json.

The authors' workbook prices one Components row for an order as

    total units = qty_per_strut x struts + qty_per_robot_extra x robots     (struts = robots x struts_per_robot)
    packs       = ceil(total units / pack size)
    price       = the price of the first tier whose bound the total stays strictly below ("<", as the workbook's
                  IF formulas compare); the last tier has no bound
    cost        = packs x price

This module carries no price, pack size, tier bound or quantity: every number comes from the price table (rows
`pack`, `tiers[].below_total_qty`, `tiers[].price`) and from the order size in grader_config.json. stdlib only.
"""
import math

__all__ = ["PriceBook", "order_units", "row_cost"]


def _as_count(x):
    """A non-negative whole number of pieces from a JSON number, else None (a fractional, negative, non-finite or
    non-numeric quantity is not a count)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if isinstance(x, float):
        if not math.isfinite(x) or not x.is_integer():
            return None
        x = int(x)
    return x if x >= 0 else None


def order_units(qty_per_strut, qty_per_robot_extra, robots, struts_per_robot):
    """Units ordered for `robots` robots: qty_per_strut x robots x struts_per_robot + qty_per_robot_extra x robots.
    A quantity that is not a whole non-negative number counts as zero pieces."""
    a = _as_count(qty_per_strut) or 0
    b = _as_count(qty_per_robot_extra) or 0
    return a * int(robots) * int(struts_per_robot) + b * int(robots)


def _packs(units, pack):
    """ceil(units / pack) for whole units; a whole-number pack size is divided exactly (integers), any other size
    with float division (the workbook's CEILING)."""
    if float(pack).is_integer():
        p = int(pack)
        return -(-int(units) // p)
    return int(math.ceil(units / float(pack)))


def row_cost(row, units):
    """Cost of `units` pieces of one price-table row under the authors' function (0 for no pieces)."""
    units = int(units)
    if units <= 0:
        return 0.0
    price = None
    for tier in row["tiers"]:
        bound = tier.get("below_total_qty")
        if bound is None or units < bound:
            price = float(tier["price"])
            break
    if price is None:                                   # a table without an unbounded last tier: the last price
        price = float(row["tiers"][-1]["price"])
    return _packs(units, row["pack"]) * price


class PriceBook:
    """The rows of gt/price_table.json by their workbook row number."""

    def __init__(self, price_table):
        self.table = price_table
        self.rows = {int(r["row"]): r for r in price_table["rows"]}

    def has(self, row_number):
        return row_number is not None and int(row_number) in self.rows

    def priced(self, row_number):
        """True when the row is in the priced scope of C5 (the table's `priced_scope`)."""
        return self.has(row_number) and bool(self.rows[int(row_number)].get("priced_scope"))

    def cost(self, row_number, units):
        return row_cost(self.rows[int(row_number)], units)

    def order_total(self, robots, struts_per_robot, scope_only=False, rows=None):
        """The authors' order of `robots` robots at the table's own per-strut quantities: {row: cost} and the sum
        (all rows, or only the priced scope). Used by the self-checks against the table's recorded totals."""
        out = {}
        for rn in sorted(self.rows if rows is None else rows):
            r = self.rows[rn]
            if scope_only and not r.get("priced_scope"):
                continue
            per_strut = r["per_strut_author"]
            per_strut = int(per_strut) if float(per_strut).is_integer() else per_strut
            out[rn] = row_cost(r, order_units(per_strut, 0, robots, struts_per_robot))
        return out, sum(out[k] for k in sorted(out))
