<?php
session_start(); 
include('../connect.php');
$drug_ids=$_POST['drug_id'];
$qtyps=$_POST['qtyp'];
$buying_prices=$_POST['buying_price'];
$selling_prices=$_POST['selling_price'];
$ins_prices=$_POST['ins_price'];

//making an array of values

$drug_id = $drug_ids; 
$buying_price = $buying_prices; 
$selling_price = $selling_prices; 
$quantity = $qtyps;
$ins_price= $ins_prices;

$combined_arrays = array();

for ($i = 0; $i < count($drug_id); $i++) {
    array_push($combined_arrays, array(
        'drug_id' => $drug_id[$i],
        'buying_price' => $buying_price[$i],
        'selling_price' => $selling_price[$i],
        'quantity' => $quantity[$i],
        'ins_price' => $ins_price[$i]
    ));
}

foreach ($combined_arrays as $row) {
    $mark_up=(1+(($row['selling_price']-$row['buying_price'])/$row['buying_price']));
    $ins_mark_up=(1+(($row['ins_price']-$row['buying_price'])/$row['buying_price']));
    $sql = "UPDATE drugs 
    SET 
    price = :buying_price,
    mark_up = :mark_up, 
    pharm_qty = :quantity,
    ins_mark_up = :ins_mark_up 
    WHERE drug_id = :drug_id";
   $result = $db->prepare($sql);
   $result->execute(array(
        ':buying_price' => $row['buying_price'],
        ':mark_up' => $mark_up,
        ':quantity' => $row['quantity'],
        ':ins_mark_up' => $ins_mark_up,
        ':drug_id' => $row['drug_id']
    ));
}

header("location: stock_take.php?message=1");
?>