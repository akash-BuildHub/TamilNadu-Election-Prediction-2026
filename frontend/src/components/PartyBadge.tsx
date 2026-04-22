import { Party, PARTY_LABELS } from "../types/prediction";

type Props = {
  party: Party;
};

export function PartyBadge({ party }: Props) {
  return <span className={`party-badge party-${party}`}>{PARTY_LABELS[party]}</span>;
}
