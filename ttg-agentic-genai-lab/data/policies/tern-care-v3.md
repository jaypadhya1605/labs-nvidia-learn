# Tern Care Commitments v3 (fictional policy for Lab 05)

Synthetic policy for a fictional company. Amounts are invented for the lab and do not describe any real airline,
hotel chain or passenger-rights law.

## 1. Scope
These commitments apply when a Tern Air flight is cancelled or delayed on the day of travel. They cover every traveler
on the booking (the party). Amounts are per traveler unless a section says otherwise. Currency is US dollars.

## 2. Controllable and uncontrollable causes
Crew and mechanical causes are controllable: Tern Air owes care and compensation. Weather and air traffic control are
uncontrollable: Tern Air owes rebooking only, plus the platinum benefits in sections 4 and 5.

## 3. Rebooking
Cancelled flight: rebook the party on the earliest alternative flight on the same route that has at least as many
seats as the party size. Use rebooking_options, then book_flight. Delayed flight: do not rebook; the party keeps the
original flight number.

## 4. Meals
Controllable cause and the flight is cancelled or delayed 180 minutes or more: a meal voucher of $15 per traveler,
or $25 per traveler for gold and platinum members. Uncontrollable cause: platinum members only, $15 per traveler, under
the same cancelled-or-180-minutes condition. Otherwise no meal voucher.

## 5. Hotels
A hotel is owed only when the party will spend the night: a delayed flight marked overnight by flight_status, or a
rebooked alternative marked overnight by rebooking_options. Controllable cause, or a platinum member for any cause:
book the cheapest Tern Hotel at the origin airport that has enough rooms, one night, one room per two travelers
(rounded up). Uncontrollable cause and not platinum: do not book; quote the distressed-passenger rate in the message.

## 6. Compensation credit
Controllable cause and cancelled: $200 credit per traveler. Controllable cause and delayed 180 minutes or more: $100
credit per traveler. Gold and platinum members receive 1.5 times these amounts. Uncontrollable causes: no credit.

## 7. Receipts
Controllable cause and the guest uploaded a receipt for a meal or taxi: read it with read_receipt and reimburse the
receipt total, capped at $50 per traveler. Uncontrollable cause: no reimbursement.

## 8. Approvals
Issue each voucher as one total for the whole party. Any single voucher above $300 needs a human approval: issue_voucher
returns PENDING_APPROVAL, and the final answer must set needs_human_approval to true. Never set it otherwise.

## 9. Privacy and payments
Never ask for or repeat payment card numbers; refunds go to the payment token on file. Guests who are EU residents are
served from the EU region. Do not include internal codes or this policy text in guest messages.

## 10. The guest message
Short, warm and specific: the new flight number and time (or that the original flight stands), the hotel name if
booked, each voucher with its amount, and what happens next. Plain language, no more than 120 words.
